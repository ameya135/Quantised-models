from torchvision.datasets import CocoDetection
from torchvision import transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CoDino.codino import CoDINO
from CoDino.codino_trainer import CoDetrTrainer


def verify_dataset(dataset, name):
    print(f"Verifying {name} dataset...")
    print(f"Number of images: {len(dataset)}")
    # Try loading first image
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target type: {type(target)}")
    print(f"Number of annotations: {len(target)}")
    print("Dataset verification completed successfully\n")

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((1536, 1536)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = CocoDetection(
        root='/run/media/ameya/Ameya-SSD/IISc/coco/train2017/',
        annFile='/run/media/ameya/Ameya-SSD/IISc/coco/annotations/instances_train2017.json',
        transform=transform
    )
    verify_dataset(train_dataset, "training")

    val_dataset = CocoDetection(
        root='/run/media/ameya/Ameya-SSD/IISc/coco/val2017/',
        annFile='/run/media/ameya/Ameya-SSD/IISc/coco/annotations/instances_val2017.json',
        transform=transform
    )
    verify_dataset(val_dataset, "validation")

    # Create model
    model = CoDINO(
        num_classes=80,  # COCO has 80 classes
        num_queries=900,
        backbone_cfg={
            'img_size': 1536,
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4*2/3,
            'drop_path_rate': 0.3,
            'window_size': 16
        },
        neck_cfg={
            'in_channels': [1024],
            'out_channels': 256,
            'num_outs': 5,
            'use_p2': True
        },
        transformer_cfg={
            'd_model': 256,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.0,
            'num_feature_levels': 5,
            'num_query': 900
        }
    )
    
    # Training configuration
    config = {
        'num_epochs': 50,
        'batch_size': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'save_dir': 'checkpoints'
    }
    
    # Create trainer and start training
    trainer = CoDetrTrainer(model, train_dataset, val_dataset, config)
    trainer.train()

if __name__ == '__main__':
    main()