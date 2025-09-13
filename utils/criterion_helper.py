import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class FocalRegionLoss(nn.Module):
    """
    Focal Region Loss from "Region-Attention-Transformer-for-Medical-Image-Restoration"
    Adapted for anomaly detection
    """
    def __init__(self, weight, beta=2.0, epsilon=1e-3, loss_type='mse'):
        super(FocalRegionLoss, self).__init__()
        self.weight = weight
        self.epsilon2 = epsilon * epsilon
        self.beta = beta
        self.loss_type = loss_type
        
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        mask = input.get("mask", None)
        
        # Resize mask to match feature size if provided
        if mask is not None:
            # mask: [B, 1, H_orig, W_orig] -> [B, H_feat, W_feat]
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.squeeze(1)  # [B, H_orig, W_orig]
            
            # Resize mask to match feature dimensions
            b, h_orig, w_orig = mask.shape
            _, _, h_feat, w_feat = feature_rec.shape
            
            if h_orig != h_feat or w_orig != w_feat:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(), 
                    size=(h_feat, w_feat), 
                    mode='nearest'
                ).squeeze(1).long()  # [B, H_feat, W_feat]
        else:
            mask = self.create_pseudo_mask(feature_rec, feature_align)
        
        return self.compute_focal_region_loss(feature_rec, feature_align, mask)
    
    def create_pseudo_mask(self, feature_rec, feature_align, n_regions=5):
        """Create pseudo mask using reconstruction error clustering"""
        b, c, h, w = feature_rec.shape
        
        if self.loss_type == 'l1':
            error = torch.abs(feature_rec - feature_align)
        else:
            error = (feature_rec - feature_align) ** 2
        
        error = torch.mean(error, dim=1)  # [B, H, W]
        
        pseudo_masks = []
        for bi in range(b):
            error_flat = error[bi].flatten().detach().cpu().numpy()
            quantiles = np.linspace(0, 1, n_regions + 1)
            thresholds = np.quantile(error_flat, quantiles)
            
            mask_bi = torch.zeros_like(error[bi])
            for i in range(n_regions):
                if i == n_regions - 1:
                    region_mask = error[bi] >= thresholds[i]
                else:
                    region_mask = (error[bi] >= thresholds[i]) & (error[bi] < thresholds[i + 1])
                mask_bi[region_mask] = i
            
            pseudo_masks.append(mask_bi)
        
        return torch.stack(pseudo_masks, dim=0)  # [B, H, W]
    
    def compute_focal_region_loss(self, pred, target, mask):
        """Compute the focal region loss following the original paper"""
        # Compute base loss
        if self.loss_type == 'l1':
            loss_metric = F.l1_loss(pred, target, reduction='none')  # [B, C, H, W]
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            loss_metric = F.mse_loss(pred, target, reduction='none')  # [B, C, H, W]
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Average across channels to get [B, H, W]
        loss_metric = torch.mean(loss_metric, dim=1)
        
        # Initialize weight same as loss_metric
        weight = loss_metric.clone().detach()
        
        b = weight.shape[0]
        for bi in range(b):
            mask_bi = mask[bi]  # [H, W]
            
            # Process each region (following original implementation)
            for cls_i in range(int(mask_bi.max().item()) + 1):
                region = (mask_bi == cls_i)
                area_i = torch.sum(region)
                
                if area_i > 0:
                    # Compute average loss for this region
                    avg_i = torch.mean(weight[bi][region])
                    # Assign this average to all pixels in the region
                    weight[bi][region] = avg_i
        
        # Normalize weights to [0, 1] (following original paper)
        weight_max = weight.max()
        if weight_max > 0:
            weight = weight / (weight_max + self.epsilon2)
        else:
            weight = weight + self.epsilon2
        weight = torch.clamp(weight, min=0.0, max=1.0)
        
        # Apply focal weighting: loss * (weight * beta + 1)
        weighted_loss = loss_metric * (weight * self.beta + 1)
        
        return torch.mean(weighted_loss)


class SpatialRegionFocalLoss(nn.Module):
    """
    Spatial Region Focal Loss with proper overlap handling
    Divides feature map into spatial regions and applies focal weighting based on region importance
    """
    def __init__(self, weight, beta=2.0, epsilon=1e-3, loss_type='mse', 
                 region_size=3, overlap=False):
        super(SpatialRegionFocalLoss, self).__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.beta = beta
        self.loss_type = loss_type
        self.region_size = region_size
        self.overlap = overlap
        
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        
        return self.compute_spatial_focal_loss(feature_rec, feature_align)
    
    def compute_spatial_focal_loss(self, pred, target):
        """
        Compute spatial focal loss with proper overlap handling
        """
        b, c, h, w = pred.shape
        
        # Compute base loss for each pixel
        if self.loss_type == 'l1':
            loss_metric = F.l1_loss(pred, target, reduction='none')  # [B, C, H, W]
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            loss_metric = F.mse_loss(pred, target, reduction='none')  # [B, C, H, W]
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        loss_metric = torch.mean(loss_metric, dim=1)  # [B, H, W]
        
        # Determine step size
        if self.overlap:
            step = max(1, self.region_size // 2)
        else:
            step = self.region_size
        
        focal_weights = torch.ones_like(loss_metric)
        
        for bi in range(b):
            loss_bi = loss_metric[bi]  # [H, W]
            
            if self.overlap:
                # Proper overlap handling with weighted average
                weight_accumulator = torch.zeros_like(loss_bi)
                count_accumulator = torch.zeros_like(loss_bi)
                
                # Collect all regions and their importance
                for start_h in range(0, h, step):
                    for start_w in range(0, w, step):
                        end_h = min(start_h + self.region_size, h)
                        end_w = min(start_w + self.region_size, w)
                        
                        # Calculate average loss for this region
                        region_loss = torch.mean(loss_bi[start_h:end_h, start_w:end_w])
                        
                        # Add to accumulator (weighted by region importance)
                        weight_accumulator[start_h:end_h, start_w:end_w] += region_loss
                        count_accumulator[start_h:end_h, start_w:end_w] += 1
                
                # Average overlapping weights
                # Each pixel gets average importance of all regions containing it
                region_importance = weight_accumulator / (count_accumulator + self.epsilon)
                
            else:
                # Non-overlap: direct assignment
                region_importance = torch.zeros_like(loss_bi)
                
                for start_h in range(0, h, step):
                    for start_w in range(0, w, step):
                        end_h = min(start_h + self.region_size, h)
                        end_w = min(start_w + self.region_size, w)
                        
                        region_loss = torch.mean(loss_bi[start_h:end_h, start_w:end_w])
                        region_importance[start_h:end_h, start_w:end_w] = region_loss
            
            # # Normalize region importance to [0, 1]
            # min_importance = torch.min(region_importance)
            # max_importance = torch.max(region_importance)
            
            # if max_importance > min_importance:
            #     normalized_importance = (region_importance - min_importance) / \
            #                           (max_importance - min_importance + self.epsilon)
            # else:
            #     normalized_importance = torch.ones_like(region_importance) * 0.5
            
            # Z-score + Sigmoid normalization (use existing region_importance tensor)
            median_loss = torch.median(region_importance)
            std_loss = torch.std(region_importance)

            if std_loss > self.epsilon:
                z_scores = (region_importance - median_loss) / std_loss
                normalized_importance = torch.sigmoid(z_scores)
            else:
                # Fallback: relative ranking 
                region_flat = region_importance.flatten()
                _, indices = torch.sort(region_flat, descending=True)
                normalized_flat = torch.zeros_like(region_flat)
                for i, idx in enumerate(indices):
                    # Higher loss gets higher score (for higher focal weight)
                    normalized_flat[idx] = (len(indices) - 1 - i) / (len(indices) - 1) if len(indices) > 1 else 0.5
                normalized_importance = normalized_flat.reshape_as(region_importance)
            
            # Apply focal weighting
            region_weights = 1.0 + self.beta * normalized_importance
            focal_weights[bi] = region_weights
        
        # Apply focal weighting to loss
        weighted_loss = loss_metric * focal_weights
        
        return torch.mean(weighted_loss)


class FeatureFocalRegionLoss(nn.Module):
    """Direct replacement for FeatureMSELoss using Focal Region Loss"""
    def __init__(self, weight, beta=1.0, epsilon=1e-3, loss_type='l1'):
        super().__init__()
        self.weight = weight
        self.focal_region_loss = FocalRegionLoss(
            weight=1.0, 
            beta=beta, 
            epsilon=epsilon, 
            loss_type=loss_type
        )
    
    def forward(self, input):
        return self.focal_region_loss(input)


class FeatureSpatialFocalRegionLoss(nn.Module):
    """Direct replacement for FeatureMSELoss using Spatial Region Focal Loss"""
    def __init__(self, weight, beta=1.0, epsilon=1e-3, loss_type='l1', 
                 region_size=3, overlap=False):
        super().__init__()
        self.weight = weight
        self.spatial_focal_loss = SpatialRegionFocalLoss(
            weight=1.0,
            beta=beta,
            epsilon=epsilon,
            loss_type=loss_type,
            region_size=region_size,
            overlap=overlap
        )
    
    def forward(self, input):
        return self.spatial_focal_loss(input)


class CombinedMSEFocalRegionLoss(nn.Module):
    """Combines MSE loss and Focal Region Loss"""
    def __init__(self, weight, mse_weight=0.4, focal_weight=0.6, 
                 beta=1.0, epsilon=1e-3, loss_type='l1'):
        super().__init__()
        self.weight = weight
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_region_loss = FocalRegionLoss(
            weight=1.0, 
            beta=beta, 
            epsilon=epsilon, 
            loss_type=loss_type
        )
    
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        
        # MSE component
        mse_component = self.mse_loss(feature_rec, feature_align)
        
        # Focal region component  
        focal_component = self.focal_region_loss(input)
        
        # Combined loss
        total_loss = self.mse_weight * mse_component + self.focal_weight * focal_component
        
        return total_loss


class CombinedMSESpatialFocalLoss(nn.Module):
    """Combines MSE loss and Spatial Focal Region Loss"""
    def __init__(self, weight, mse_weight=0.4, spatial_focal_weight=0.6, 
                 beta=1.0, epsilon=1e-3, loss_type='l1', region_size=3, overlap=False):
        super().__init__()
        self.weight = weight
        self.mse_weight = mse_weight
        self.spatial_focal_weight = spatial_focal_weight
        
        self.mse_loss = nn.MSELoss()
        self.spatial_focal_loss = SpatialRegionFocalLoss(
            weight=1.0,
            beta=beta,
            epsilon=epsilon,
            loss_type=loss_type,
            region_size=region_size,
            overlap=overlap
        )
    
    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        
        # MSE component
        mse_component = self.mse_loss(feature_rec, feature_align)
        
        # Spatial focal component
        spatial_focal_component = self.spatial_focal_loss(input)
        
        # Combined loss
        total_loss = self.mse_weight * mse_component + self.spatial_focal_weight * spatial_focal_component
        
        return total_loss


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""
    
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
