import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class CoDetrTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: CocoDetection,
        val_dataset: CocoDetection,
        config: dict
    ):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training configuration
        self.num_epochs = config.get('num_epochs', 50)
        self.batch_size = config.get('batch_size', 2)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.num_workers = config.get('num_workers', 4)
        self.save_dir = config.get('save_dir', 'checkpoints')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs
        )
        
        # Loss functions
        self.criterion_bbox = nn.L1Loss()
        self.criterion_giou = self.generalized_iou_loss
        self.criterion_cls = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join('runs', time.strftime("%Y%m%d-%H%M%S")))
        
        # Create checkpoint directory
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            
            # Convert COCO annotations to the required format
            boxes = []
            labels = []
            for ann in target:
                bbox = ann['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                boxes.append([
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3]
                ])
                labels.append(ann['category_id'])
            
            targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            })
            
        return torch.stack(images, 0), targets

    def generalized_iou_loss(self, boxes1, boxes2):
        # Implementation of GIoU loss
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        lt = torch.max(boxes1[..., :2], boxes2[..., :2])
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        union = area1 + area2 - inter
        iou = inter / union
        
        lt = torch.min(boxes1[..., :2], boxes2[..., :2])
        rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        enclosed_area = wh[..., 0] * wh[..., 1]
        
        giou = iou - (enclosed_area - union) / enclosed_area
        return 1 - giou.mean()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            # Move targets to device
            targets = [{
                'boxes': t['boxes'].to(self.device),
                'labels': t['labels'].to(self.device)
            } for t in targets]            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            # Calculate losses
            loss_bbox = self.criterion_bbox(outputs['pred_boxes'], torch.stack([t['boxes'] for t in targets]))
            loss_giou = self.criterion_giou(outputs['pred_boxes'], torch.stack([t['boxes'] for t in targets]))
            loss_cls = self.criterion_cls(outputs['pred_logits'], torch.stack([t['labels'] for t in targets]))
            
            loss = loss_bbox + loss_giou + loss_cls
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
            # Log to TensorBoard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train', loss.item(), step)
            self.writer.add_scalar('Loss/bbox', loss_bbox.item(), step)
            self.writer.add_scalar('Loss/giou', loss_giou.item(), step)
            self.writer.add_scalar('Loss/cls', loss_cls.item(), step)
        
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(self.val_loader):
            images = images.to(self.device)
            # Convert targets to the correct format
            targets = [{k: torch.tensor(v).to(self.device) for k, v in t.items()} for t in targets]
            
            outputs = self.model(images)
            
            # Calculate losses
            loss_bbox = self.criterion_bbox(outputs['pred_boxes'], torch.stack([t['boxes'] for t in targets]))
            loss_giou = self.criterion_giou(outputs['pred_boxes'], torch.stack([t['boxes'] for t in targets]))
            loss_cls = self.criterion_cls(outputs['pred_logits'], torch.stack([t['labels'] for t in targets]))
            
            loss = loss_bbox + loss_giou + loss_cls
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return avg_loss

    def train(self):
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs}')
            
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save checkpoint if validation loss improved
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': best_loss,
                }
                torch.save(
                    checkpoint,
                    os.path.join(self.save_dir, f'checkpoint_best.pth')
                )
            
            # Save latest checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': best_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(self.save_dir, f'checkpoint_latest.pth')
            )
        
        self.writer.close()