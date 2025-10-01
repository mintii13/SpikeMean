import torch
import torch.nn as nn
from copy import deepcopy
import os
import logging

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

from spikingjelly.activation_based import layer, neuron
from models.initializer import initialize_from_cfg

logger = logging.getLogger("global_logger")

__all__ = ['SpikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, outlayers, outstrides, frozen_layers=[], 
                 num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None, 
                 spiking_neuron: callable = None, initializer=None, **kwargs):
        super(SpikingResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.outlayers = outlayers
        self.outstrides = outstrides
        self.frozen_layers = frozen_layers
        
        # Calculate output planes
        layer_outplanes = [64] + [i * block.expansion for i in [64, 128, 256, 512]]
        layer_outplanes = list(map(int, layer_outplanes))
        self.outplanes = [layer_outplanes[i] for i in outlayers]
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial layers
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       spiking_neuron=spiking_neuron, **kwargs)
        
        # Classification layers (not used in feature extraction but kept for compatibility)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
        initialize_from_cfg(self, initializer)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, 
                    spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def get_outplanes(self):
        """Get dimension of the output tensor"""
        return self.outplanes
    
    def get_outstrides(self):
        """Get strides of the output tensor"""
        return self.outstrides
        
    @property
    def layer0(self):
        return nn.Sequential(self.conv1, self.bn1, self.sn1, self.maxpool)

    def forward(self, input):
        """Forward pass matching ResNet interface"""
        x = input["image"]
        
        outs = []
        for layer_idx in range(0, 5):
            layer = getattr(self, f"layer{layer_idx}", None)
            if layer is not None:
                x = layer(x)
                outs.append(x)
        
        features = [outs[i] for i in self.outlayers]
        return {"features": features, "strides": self.get_outstrides()}

    def freeze_layer(self):
        """Freeze specified layers"""
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.sn1, self.maxpool),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for layer_idx in self.frozen_layers:
            layer = layers[layer_idx]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Set training mode with frozen layer support"""
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.freeze_layer()
        return self
    
    def reset_spiking_neurons(self):
        """Reset all spiking neuron states"""
        from spikingjelly.activation_based import functional
        functional.reset_net(self)


def _spiking_resnet(arch, block, layers, pretrained, pretrained_model, progress, 
                    spiking_neuron, **kwargs):
    """Build spiking resnet with optional pretrained weights"""
    # Parse string neuron type
    if isinstance(spiking_neuron, str):
        spiking_neuron = getattr(neuron, spiking_neuron)
        logger.info(f"Using spiking neuron: {spiking_neuron}")
    elif spiking_neuron is None:
        spiking_neuron = neuron.IFNode
        logger.info("Using default spiking neuron: IFNode")
    
    model = SpikingResNet(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    
    if pretrained:
        if pretrained_model and os.path.exists(pretrained_model):
            state_dict = torch.load(pretrained_model, map_location='cpu')
            logger.info(f"Loading pretrained {arch} from {pretrained_model}")
        else:
            if pretrained_model:
                logger.warning(f"Pretrained model not found: {pretrained_model}, loading from URL")
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            logger.info(f"Loading pretrained {arch} from {model_urls[arch]}")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Load pretrained {arch}\nmissing_keys: {missing_keys}\nunexpected_keys: {unexpected_keys}")
    
    return model


def spiking_resnet18(pretrained=False, pretrained_model=None, progress=True, 
                     spiking_neuron: callable=None, **kwargs):
    """Spiking ResNet-18"""
    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnet34(pretrained=False, pretrained_model=None, progress=True, 
                     spiking_neuron: callable=None, **kwargs):
    """Spiking ResNet-34"""
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnet50(pretrained=False, pretrained_model=None, progress=True, 
                     spiking_neuron: callable=None, **kwargs):
    """Spiking ResNet-50"""
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnet101(pretrained=False, pretrained_model=None, progress=True, 
                      spiking_neuron: callable=None, **kwargs):
    """Spiking ResNet-101"""
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnet152(pretrained=False, pretrained_model=None, progress=True, 
                      spiking_neuron: callable=None, **kwargs):
    """Spiking ResNet-152"""
    return _spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnext50_32x4d(pretrained=False, pretrained_model=None, progress=True, 
                            spiking_neuron: callable=None, **kwargs):
    """Spiking ResNeXt-50 32x4d"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_resnext101_32x8d(pretrained=False, pretrained_model=None, progress=True, 
                             spiking_neuron: callable=None, **kwargs):
    """Spiking ResNeXt-101 32x8d"""
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_wide_resnet50_2(pretrained=False, pretrained_model=None, progress=True, 
                            spiking_neuron: callable=None, **kwargs):
    """Spiking Wide ResNet-50-2"""
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)


def spiking_wide_resnet101_2(pretrained=False, pretrained_model=None, progress=True, 
                             spiking_neuron: callable=None, **kwargs):
    """Spiking Wide ResNet-101-2"""
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, 
                          pretrained_model, progress, spiking_neuron, **kwargs)