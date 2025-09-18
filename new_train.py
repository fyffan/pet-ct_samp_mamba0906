#!/usr/bin/env python3
"""
Training script for the new weakly supervised PET-CT tumor segmentation model
with SAM integration.

This script adapts the original CIPA dataset format to work with the new model
that requires separate PET and CT inputs and supports bounding box annotations.
"""

import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import argparse
from easydict import EasyDict as edict
import json
import logging
from typing import List, Dict, Tuple, Optional

# Import the new model
from new_model import build_model_with_sam, calculate_metrics

# Import original utilities that we can reuse
try:
    from utils.image_augmentation import randomShiftScaleRotate, randomHorizontalFlip, randomcrop
    from utils.logger import get_logger
except ImportError:
    print("Warning: Some utility modules not found. Using fallback implementations.")


class AdaptedPETCTDataset(Dataset):
    """
    Adapted dataset for the new weakly supervised model.

    This dataset loads PET and CT images separately and handles both
    full masks and bounding box annotations for weak supervision.
    """

    def __init__(self, image_list: List[str], img_root: str, transforms: bool = False,
                 use_weak_supervision: bool = True, generate_boxes: bool = True):
        super().__init__()
        self.image_list = image_list
        self.img_root = img_root
        self.transforms = transforms
        self.use_weak_supervision = use_weak_supervision
        self.generate_boxes = generate_boxes

        # File suffixes
        self.pet_suffix = "_PET.png"
        self.ct_suffix = "_CT.png"
        self.mask_suffix = "_mask.png"

        print(f"Dataset initialized with {len(image_list)} samples")
        print(f"Transforms: {transforms}, Weak supervision: {use_weak_supervision}")

    def _read_data(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read PET, CT images and mask"""
        patient_id = image_id.split("_")[0]

        pet_path = os.path.join(self.img_root, patient_id, f"{image_id}{self.pet_suffix}")
        ct_path = os.path.join(self.img_root, patient_id, f"{image_id}{self.ct_suffix}")
        mask_path = os.path.join(self.img_root, patient_id, f"{image_id}{self.mask_suffix}")

        # Read images
        pet = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Assertions
        assert pet is not None, f"PET image not found: {pet_path}"
        assert ct is not None, f"CT image not found: {ct_path}"
        assert mask is not None, f"Mask not found: {mask_path}"

        return pet, ct, mask

    def _generate_bounding_box(self, mask: np.ndarray) -> Optional[List[float]]:
        """Generate bounding box from mask for weak supervision"""
        if mask.max() == 0:
            return None

        # Find contours
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Return in (x1, y1, x2, y2) format
        return [float(x), float(y), float(x + w), float(y + h)]

    def _apply_augmentations(self, pet: np.ndarray, ct: np.ndarray,
                           mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentations"""
        if not self.transforms:
            return pet, ct, mask

        # Ensure 3D arrays
        if pet.ndim == 2:
            pet = np.expand_dims(pet, axis=2)
        if ct.ndim == 2:
            ct = np.expand_dims(ct, axis=2)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)

        try:
            # Apply augmentations to concatenated image for consistency
            combined_img = np.concatenate([pet, ct], axis=2)
            combined_img, mask = randomShiftScaleRotate(combined_img, mask)
            combined_img, mask = randomHorizontalFlip(combined_img, mask)
            combined_img, mask = randomcrop(combined_img, mask)

            # Split back
            pet = combined_img[:, :, :1]
            ct = combined_img[:, :, 1:]

        except Exception as e:
            print(f"Augmentation failed: {e}, using original images")
            # Keep original images if augmentation fails
            pass

        return pet, ct, mask

    def _normalize_images(self, pet: np.ndarray, ct: np.ndarray,
                         mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize images to tensors"""

        # Ensure correct dimensions
        if pet.ndim == 2:
            pet = np.expand_dims(pet, axis=2)
        if ct.ndim == 2:
            ct = np.expand_dims(ct, axis=2)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)

        # Normalize images: [0, 255] -> [0, 1]
        pet = pet.astype(np.float32) / 255.0
        ct = ct.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors and adjust dimensions: H,W,C -> C,H,W
        pet = torch.from_numpy(pet.transpose(2, 0, 1))
        ct = torch.from_numpy(ct.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.transpose(2, 0, 1))

        # Binarize mask
        mask = (mask >= 0.5).float()

        return pet, ct, mask

    def __getitem__(self, index: int) -> Dict:
        """Get a single sample"""
        image_id = self.image_list[index]

        # Read data
        pet, ct, mask = self._read_data(image_id)

        # Apply augmentations
        pet, ct, mask = self._apply_augmentations(pet, ct, mask)

        # Generate bounding box if needed
        box = None
        if self.generate_boxes:
            box = self._generate_bounding_box(mask.squeeze() if mask.ndim == 3 else mask)

        # Normalize
        pet, ct, mask = self._normalize_images(pet, ct, mask)

        sample = {
            'pet': pet,
            'ct': ct,
            'mask': mask,
            'image_id': image_id
        }

        if box is not None:
            sample['box'] = torch.tensor(box, dtype=torch.float32)

        return sample

    def __len__(self) -> int:
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        pets = torch.stack([item['pet'] for item in batch])
        cts = torch.stack([item['ct'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])

        batch_data = {
            'pet': pets,
            'ct': cts,
            'masks': masks,
            'image_ids': [item['image_id'] for item in batch]
        }

        # Handle boxes
        if 'box' in batch[0]:
            boxes = torch.stack([item['box'] for item in batch])
            batch_data['boxes'] = boxes

        return batch_data


def prepare_adapted_dataset(data_root: str, split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Prepare train/test splits for the adapted dataset
    """
    train_file = os.path.join(data_root, 'train.txt')
    test_file = os.path.join(data_root, 'test.txt')

    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Using existing train/test split files")
        with open(train_file, 'r') as f:
            train_list = [x.strip() for x in f.readlines() if x.strip()]
        with open(test_file, 'r') as f:
            test_list = [x.strip() for x in f.readlines() if x.strip()]
    else:
        print("Split files not found. Please ensure train.txt and test.txt exist in the data directory.")
        print(f"Expected files: {train_file}, {test_file}")
        train_list, test_list = [], []

    return train_list, test_list


def setup_logging(save_dir: str) -> logging.Logger:
    """Setup logging configuration"""
    log_file = os.path.join(save_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                   data_loader: DataLoader, device: torch.device,
                   epoch: int, print_freq: int = 50) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, batch_data in enumerate(data_loader):
        pet = batch_data['pet'].to(device)
        ct = batch_data['ct'].to(device)
        masks = batch_data['masks'].to(device)

        boxes = batch_data.get('boxes', None)
        if boxes is not None:
            boxes = boxes.to(device)

        optimizer.zero_grad()

        # Forward pass
        try:
            outputs = model(pet, ct, boxes=boxes, gt_masks=masks)

            # Calculate total loss
            losses = outputs['losses']
            total_batch_loss = sum(losses.values())

            # Backward pass
            total_batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += total_batch_loss.item()

            if batch_idx % print_freq == 0:
                print(f'Epoch [{epoch}], Batch [{batch_idx}/{num_batches}], '
                      f'Loss: {total_batch_loss.item():.4f}')
                for loss_name, loss_value in losses.items():
                    print(f'  {loss_name}: {loss_value.item():.4f}')

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / max(num_batches, 1)


def evaluate_model(model: nn.Module, data_loader: DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}
    num_samples = 0

    with torch.no_grad():
        for batch_data in data_loader:
            pet = batch_data['pet'].to(device)
            ct = batch_data['ct'].to(device)
            masks = batch_data['masks'].to(device)

            # Forward pass
            try:
                outputs = model(pet, ct)
                pred_masks = outputs['final_mask']

                # Calculate metrics for each sample in batch
                for i in range(pred_masks.shape[0]):
                    metrics = calculate_metrics(pred_masks[i], masks[i])
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                    num_samples += 1

            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue

    # Average metrics
    if num_samples > 0:
        for key in total_metrics:
            total_metrics[key] /= num_samples

    return total_metrics


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int, metrics: Dict[str, float],
                   save_path: str, is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='New Weakly Supervised PET-CT Tumor Segmentation Training')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data/PCLT20k/',
                       help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    # Model arguments
    parser.add_argument('--sam_type', type=str, default='medsam',
                       choices=['standard_sam_vit_b', 'standard_sam_vit_l', 'standard_sam_vit_h',
                               'medsam', 'simplified_sam'],
                       help='Type of SAM model to use')
    parser.add_argument('--sam_checkpoint', type=str, default='',
                       help='Path to SAM checkpoint')
    parser.add_argument('--use_weak_supervision', action='store_true', default=True,
                       help='Use weak supervision with bounding boxes')

    # Save and resume arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints_new/',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency (epochs)')
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency (batches)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"experiment_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(save_dir)
    logger.info(f"Starting training with arguments: {args}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Prepare datasets
    train_list, test_list = prepare_adapted_dataset(args.data_root)
    logger.info(f'Train samples: {len(train_list)}, Test samples: {len(test_list)}')

    if len(train_list) == 0:
        logger.error("No training data found! Please check your data directory and split files.")
        return

    # Create datasets
    train_dataset = AdaptedPETCTDataset(
        train_list, args.data_root,
        transforms=True,
        use_weak_supervision=args.use_weak_supervision,
        generate_boxes=args.use_weak_supervision
    )

    val_dataset = AdaptedPETCTDataset(
        test_list, args.data_root,
        transforms=False,
        use_weak_supervision=args.use_weak_supervision,
        generate_boxes=args.use_weak_supervision
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=AdaptedPETCTDataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AdaptedPETCTDataset.collate_fn,
        pin_memory=True
    )

    # Create model
    logger.info(f"Building model with SAM type: {args.sam_type}")

    # Determine SAM checkpoint path
    sam_checkpoint_path = args.sam_checkpoint
    if not sam_checkpoint_path and args.sam_type != 'simplified_sam':
        default_paths = {
            'standard_sam_vit_b': './checkpoints/sam_vit_b_01ec64.pth',
            'standard_sam_vit_l': './checkpoints/sam_vit_l_0b3195.pth',
            'standard_sam_vit_h': './checkpoints/sam_vit_h_4b8939.pth',
            'medsam': './checkpoints/medsam_vit_b.pth'
        }
        sam_checkpoint_path = default_paths.get(args.sam_type)

    # Check if checkpoint exists
    if args.sam_type != 'simplified_sam' and sam_checkpoint_path:
        if not os.path.exists(sam_checkpoint_path):
            logger.warning(f"SAM checkpoint not found at {sam_checkpoint_path}")
            logger.warning("Please download the appropriate SAM model first.")
            logger.info("Falling back to simplified SAM implementation.")
            args.sam_type = 'simplified_sam'
            sam_checkpoint_path = None

    try:
        model = build_model_with_sam(args.sam_type, sam_checkpoint_path)
        model.to(device)

        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'Total trainable parameters: {total_params:,}')

    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        logger.info("Falling back to simplified SAM")
        model = build_model_with_sam('simplified_sam', None)
        model.to(device)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint.get('metrics', {}).get('dice', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # Save config
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Training loop
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.print_freq)
        scheduler.step()

        logger.info(f'Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            metrics = evaluate_model(model, val_loader, device)
            logger.info(f'Validation - IoU: {metrics["iou"]:.4f}, Dice: {metrics["dice"]:.4f}, Acc: {metrics["accuracy"]:.4f}')

            # Save best model
            is_best = metrics['dice'] > best_dice
            if is_best:
                best_dice = metrics['dice']
                logger.info(f'New best model saved with Dice: {best_dice:.4f}')

            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0 or is_best:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, save_path, is_best)

    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, device)
    logger.info(f'Final Results - IoU: {final_metrics["iou"]:.4f}, Dice: {final_metrics["dice"]:.4f}, Acc: {final_metrics["accuracy"]:.4f}')

    total_time = time.time() - start_time
    logger.info(f'Training completed in {total_time/3600:.2f} hours')
    logger.info(f'Best Dice score: {best_dice:.4f}')


if __name__ == "__main__":
    main()