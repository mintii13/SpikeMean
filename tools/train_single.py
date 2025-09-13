#!/usr/bin/env python3
"""
Direct training script for all MVTec classes
Usage: python tools/train_single.py
"""

import os
import sys
import yaml
import copy
import subprocess
import time

# Setup paths
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to IAD root
os.chdir(current_dir)
sys.path.insert(0, current_dir)
os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + os.pathsep + current_dir

def train_single_class(cls_name, base_config):
    """Train a single class"""
    try:
        # Create config for this class
        cls_config = copy.deepcopy(base_config)
        cls_config['wandb']['name'] = f"{cls_name}_zscore_sigmol_500_256_256_0mse_1spatial_mse_3non_beta2"
        cls_config['wandb']['project'] = "UniAD-MVTec-single"
        
        # Save temporary config file
        config_path = f"config_{cls_name}_temp.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(cls_config, f, default_flow_style=False)
        
        print(f"Starting training for {cls_name}")
        
        # Run training
        cmd = [
            sys.executable, "-u", "tools/train_val.py",
            "--config", config_path,
            "--class_name", cls_name,
            "--single_gpu"
        ]
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print real-time output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[{cls_name}] {output.strip()}")
        
        return_code = process.poll()
        
        # Clean up
        try:
            os.remove(config_path)
        except:
            pass
        
        if return_code == 0:
            print(f"{cls_name}: Training completed successfully")
            return True
        else:
            print(f"{cls_name}: Training failed")
            return False
            
    except Exception as e:
        print(f"{cls_name}: Exception - {str(e)}")
        return False

def main():
    """Main training function"""
    print("UniAD Single-Class Training")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('tools/train_val.py'):
        print("Error: tools/train_val.py not found!")
        print("Please run this script from the IAD project root directory")
        sys.exit(1)
    
    # Load base config from tools/ directory
    config_path = os.path.join('tools', 'config.yaml')
    if not os.path.exists(config_path):
        print("Error: tools/config.yaml not found!")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        print("Loaded tools/config.yaml")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Class list
    class_name_list = [
        "bottle"
        # , "cable", "capsule", "hazelnut", "metal_nut",
        # "pill", "screw", "toothbrush", "transistor", "zipper",
        # "carpet", "grid", "leather", "tile", "wood"
        # 'toothbrush'
    ]
    
    print(f"Training {len(class_name_list)} classes sequentially")
    
    # Start training
    start_time = time.time()
    success_count = 0
    
    for i, cls_name in enumerate(class_name_list):
        print(f"\n{'='*60}")
        print(f"Training class {i+1}/{len(class_name_list)}: {cls_name}")
        print(f"{'='*60}")
        
        success = train_single_class(cls_name, base_config)
        if success:
            success_count += 1
        
        # Rest between classes (except last one)
        if i < len(class_name_list) - 1:
            print(f"Resting 30 seconds before next class...")
            time.sleep(30)
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(class_name_list)} classes")
    print(f"Total time: {duration/3600:.1f} hours ({duration/60:.1f} minutes)")
    print(f"Average per class: {duration/len(class_name_list)/60:.1f} minutes")
    
    if success_count == len(class_name_list):
        print("All classes completed successfully!")
    else:
        failed = len(class_name_list) - success_count
        print(f"{failed} classes failed")
    
    print("Training completed!")

if __name__ == "__main__":
    main()