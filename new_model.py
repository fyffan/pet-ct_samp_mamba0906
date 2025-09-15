"""
Cross-Modal Interactive Perception Network with Mamba for Weakly Supervised Tumor Segmentation
Based on the structure diagram and reference papers for implementing a weakly supervised learning approach
with SAM assistance for PET-CT tumor segmentation.

Key Components:
1. Encoder: Uses existing VMamba encoder to extract multi-scale PET and CT features
2. Transformer Feature Interaction: Processes encoder outputs with learnable queries
3. CAM Generation: Creates class activation maps for both modalities
4. SAM Integration: Uses SAM for weakly supervised mask generation with box and point prompts
5. Loss Computation: Implements hard and soft label losses with weighted supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple
from models.encoders.vmamba import Backbone_VSSM


class LearnableQueries(nn.Module):
    """
    Learnable queries for each modality to interact with multi-scale features
    """
    def __init__(self, num_queries: int, embed_dim: int, num_scales: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_scales = num_scales

        # Learnable queries for each scale and modality
        self.pet_queries = nn.Parameter(torch.randn(num_scales, num_queries, embed_dim))
        self.ct_queries = nn.Parameter(torch.randn(num_scales, num_queries, embed_dim))

        # Initialize queries
        nn.init.normal_(self.pet_queries, std=0.02)
        nn.init.normal_(self.ct_queries, std=0.02)

    def forward(self):
        return self.pet_queries, self.ct_queries


class TransformerFeatureInteraction(nn.Module):
    """
    Transformer-based feature interaction module for processing multi-scale features
    with learnable queries as described in the structure diagram
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 3, num_queries: int = 100):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Learnable queries，返回的是两组queries
        self.queries = LearnableQueries(num_queries, embed_dim)

        # Self-attention for query features
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Cross-attention for query-feature interaction
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Feature projection
        self.feature_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, scale_idx: int, modality: str = 'pet'):
        """
        Args:
            features: Input features [B, C, H, W]
            scale_idx: Scale index (0-3 for different scales)
            modality: 'pet' or 'ct'
        Returns:
            query_features: Enhanced query features
        """
        B, C, H, W = features.shape

        # Get learnable queries
        pet_queries, ct_queries = self.queries()
        queries = pet_queries[scale_idx] if modality == 'pet' else ct_queries[scale_idx]
        queries = queries.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, embed_dim]

        # Reshape features to sequence format
        feat_seq = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        feat_seq = self.feature_proj(feat_seq)

        # Cross-attention between queries and features
        query_features, _ = self.cross_attention(
            query=queries,
            key=feat_seq,
            value=feat_seq
        )

        # Self-attention on query features
        query_features, _ = self.self_attention(
            query=query_features,
            key=query_features,
            value=query_features
        )

        # Apply transformer layers
        query_features = self.transformer(query_features)
        query_features = self.norm(query_features)

        return query_features


class CrossModalInteraction(nn.Module):
    """
    Cross-modal interaction module where query features from one modality
    interact with original features from another modality
    特征交互部分
    """
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, query_features: torch.Tensor, target_features: torch.Tensor):
        """
        Args:
            query_features: Query features from one modality [B, num_queries, embed_dim]
            target_features: Original features from another modality [B, C, H, W]
        Returns:
            interaction_features: Features after cross-modal interaction
        """
        B, C, H, W = target_features.shape

        # Reshape target features
        target_seq = target_features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Cross-attention
        interaction_features, _ = self.cross_attention(
            query=query_features,
            key=target_seq,
            value=target_seq
        )

        # Add residual connection and apply FFN
        interaction_features = self.norm(interaction_features + query_features)
        ffn_out = self.ffn(interaction_features)
        interaction_features = self.norm(interaction_features + ffn_out)

        return interaction_features


class CAMGenerator(nn.Module):
    """
    Class Activation Map (CAM) generator for tumor classification and localization
    """
    def __init__(self, embed_dim: int, num_classes: int = 2, num_queries: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Feature weight generation for CAM
        self.cam_conv = nn.Conv1d(embed_dim, num_classes, kernel_size=1)

    def forward(self, query_features: torch.Tensor):
        """
        Args:
            query_features: Query features [B, num_queries, embed_dim]
        Returns:
            cam: Class activation map
            classification: Classification logits
        """
        B, N, C = query_features.shape

        # Global classification
        pooled_features = self.gap(query_features.transpose(1, 2)).squeeze(-1)  # [B, C]
        classification = self.classifier(pooled_features)

        # Generate CAM
        query_features_t = query_features.transpose(1, 2)  # [B, C, N]
        cam_weights = self.cam_conv(query_features_t)  # [B, num_classes, N]

        # Apply softmax to get attention weights
        cam = F.softmax(cam_weights, dim=-1)

        return cam, classification


class PointExtractor(nn.Module):
    """
    Extract high-response points from CAM for SAM point prompts
    """
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, cam: torch.Tensor, features_shape: Tuple[int, int, int, int]):
        """
        Args:
            cam: Class activation map [B, num_classes, num_queries]
            features_shape: Original feature shape (B, C, H, W)
        Returns:
            points: High-response point coordinates
        """
        B, num_classes, num_queries = cam.shape
        _, _, H, W = features_shape

        # Take tumor class (class 1) activation
        tumor_cam = cam[:, 1, :]  # [B, num_queries]

        # Get top-k points
        _, top_indices = torch.topk(tumor_cam, self.top_k, dim=1)  # [B, top_k]

        # Convert to spatial coordinates (assuming uniform query distribution)
        queries_per_dim = int(math.sqrt(num_queries))
        points = []

        for b in range(B):
            batch_points = []
            for k in range(self.top_k):
                idx = top_indices[b, k].item()
                y = idx // queries_per_dim
                x = idx % queries_per_dim

                # Scale to feature map size
                scaled_x = int(x * W / queries_per_dim)
                scaled_y = int(y * H / queries_per_dim)

                batch_points.append([scaled_x, scaled_y])
            points.append(batch_points)

        return points


class SAMPromptEncoder(nn.Module):
    """
    Simplified SAM prompt encoder for processing box and point prompts
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Point embedding
        self.point_embeddings = nn.Embedding(2, embed_dim)  # positive/negative

        # Box embedding
        self.box_embedding = nn.Embedding(1, embed_dim)

        # Positional encoding
        self.pe_layer = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, points: Optional[List] = None, boxes: Optional[torch.Tensor] = None):
        """
        Args:
            points: List of point coordinates
            boxes: Box coordinates [B, 4] (x1, y1, x2, y2)
        Returns:
            prompt_embeddings: Encoded prompt embeddings
        """
        embeddings = []

        if boxes is not None:
            B = boxes.shape[0]
            box_embed = self.box_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

            # Add positional information
            box_centers = torch.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                (boxes[:, 1] + boxes[:, 3]) / 2
            ], dim=1)  # [B, 2]

            pos_embed = self.pe_layer(box_centers).unsqueeze(1)
            box_embed = box_embed + pos_embed
            embeddings.append(box_embed)

        if points is not None:
            # Convert points to tensors and embed
            for batch_points in points:
                point_tensor = torch.tensor(batch_points, dtype=torch.float32, device=self.point_embeddings.weight.device)
                pos_embed = self.pe_layer(point_tensor).unsqueeze(0)
                point_embed = self.point_embeddings.weight[0].unsqueeze(0).unsqueeze(0).repeat(1, len(batch_points), 1)
                point_embed = point_embed + pos_embed
                embeddings.append(point_embed)

        if embeddings:
            return torch.cat(embeddings, dim=1)
        else:
            return None


class SAMMaskDecoder(nn.Module):
    """
    Simplified SAM mask decoder for generating segmentation masks
    """
    def __init__(self, feature_dim: int = 256, embed_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        # Feature projection
        self.feature_proj = nn.Conv2d(feature_dim, embed_dim, 1)

        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, 1, 1)
        )

        # Cross-attention for prompt-feature interaction
        self.cross_attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

    def forward(self, features: torch.Tensor, prompt_embeddings: torch.Tensor):
        """
        Args:
            features: Feature map [B, C, H, W]
            prompt_embeddings: Prompt embeddings [B, num_prompts, embed_dim]
        Returns:
            masks: Predicted masks [B, 1, H, W]
        """
        B, C, H, W = features.shape

        # Project features
        proj_features = self.feature_proj(features)  # [B, embed_dim, H, W]

        # Reshape for attention
        feat_seq = proj_features.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        # Cross-attention between prompts and features
        attended_features, _ = self.cross_attention(
            query=feat_seq,
            key=prompt_embeddings,
            value=prompt_embeddings
        )

        # Reshape back
        attended_features = attended_features.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Generate masks
        masks = self.mask_head(attended_features)

        return torch.sigmoid(masks)


class WeaklySupervisedLoss(nn.Module):
    """
    Implements the weakly supervised loss with hard and soft labels
    """
    def __init__(self, hard_weight: float = 1.0, soft_weight_range: Tuple[float, float] = (0.3, 0.7)):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight_range = soft_weight_range
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_mask: torch.Tensor, box_mask: torch.Tensor, point_mask: torch.Tensor):
        """
        Args:
            pred_mask: Predicted segmentation mask [B, 1, H, W]
            box_mask: Box-generated mask [B, 1, H, W]
            point_mask: Point-generated mask [B, 1, H, W]
        Returns:
            loss: Combined hard and soft label loss
        """
        # Hard labels: intersection of box and point masks
        hard_labels = (box_mask * point_mask).float()

        # Soft labels: union minus intersection
        union_mask = torch.clamp(box_mask + point_mask, 0, 1)
        soft_labels = (union_mask - hard_labels).float()

        # Hard label loss
        hard_loss = self.bce_loss(pred_mask, hard_labels)
        hard_loss = (hard_loss * hard_labels * self.hard_weight).sum() / (hard_labels.sum() + 1e-8)

        # Soft label loss with random weights
        B = pred_mask.shape[0]
        soft_weights = torch.rand(B, 1, 1, 1, device=pred_mask.device) * \
                      (self.soft_weight_range[1] - self.soft_weight_range[0]) + self.soft_weight_range[0]

        soft_loss = self.bce_loss(pred_mask, soft_labels)
        soft_loss = (soft_loss * soft_labels * soft_weights).sum() / (soft_labels.sum() + 1e-8)

        return hard_loss + soft_loss


class NewWeaklySupervised_PETCT_Model(nn.Module):
    """
    Complete model implementing the weakly supervised PET-CT tumor segmentation
    as described in the structure diagram
    """
    def __init__(
        self,
        encoder_dims: List[int] = [96, 192, 384, 768],
        transformer_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 2,
        sam_feature_dim: int = 256
    ):
        super().__init__()

        # PET and CT Encoders (shared weights)
        self.pet_encoder = Backbone_VSSM(
            patch_size=4,
            in_chans=1,  # Single channel for PET
            dims=encoder_dims,
            depths=[2, 2, 9, 2],
            out_indices=(0, 1, 2, 3)
        )

        self.ct_encoder = Backbone_VSSM(
            patch_size=4,
            in_chans=1,  # Single channel for CT
            dims=encoder_dims,
            depths=[2, 2, 9, 2],
            out_indices=(0, 1, 2, 3)
        )

        # Feature dimension alignment
        self.feature_projections = nn.ModuleList([
            nn.Conv2d(dim, transformer_dim, 1) for dim in encoder_dims
        ])

        # Transformer feature interaction
        self.transformer_interaction = TransformerFeatureInteraction(
            embed_dim=transformer_dim,
            num_queries=num_queries
        )

        # Cross-modal interaction
        self.cross_modal_pet2ct = CrossModalInteraction(transformer_dim)
        self.cross_modal_ct2pet = CrossModalInteraction(transformer_dim)

        # CAM generators
        self.pet_cam_generator = CAMGenerator(transformer_dim, num_classes, num_queries)
        self.ct_cam_generator = CAMGenerator(transformer_dim, num_classes, num_queries)

        # Point extractor
        self.point_extractor = PointExtractor(top_k=5)

        # SAM components
        self.sam_prompt_encoder = SAMPromptEncoder(embed_dim=transformer_dim)
        self.sam_mask_decoder = SAMMaskDecoder(feature_dim=transformer_dim, embed_dim=transformer_dim)

        # Final decoder (using existing implementation)
        try:
            from models.decoders.MambaDecoder import MambaDecoderV2
            self.final_decoder = MambaDecoderV2(
                encoder_dims=encoder_dims,
                decoder_dim=transformer_dim
            )
        except ImportError:
            # Fallback simple decoder if MambaDecoderV2 is not available
            self.final_decoder = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(transformer_dim // 2, transformer_dim // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(transformer_dim // 4, 1, 1)
            )

        # Loss functions
        self.weakly_supervised_loss = WeaklySupervisedLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCEWithLogitsLoss()

    def forward(self, pet_img: torch.Tensor, ct_img: torch.Tensor,
                boxes: Optional[torch.Tensor] = None, gt_masks: Optional[torch.Tensor] = None):
        """
        Args:
            pet_img: PET image [B, 1, H, W]
            ct_img: CT image [B, 1, H, W]
            boxes: Tumor bounding boxes [B, 4] for training
            gt_masks: Ground truth masks [B, 1, H, W] for training
        Returns:
            outputs: Dictionary containing all outputs and losses
        """
        B = pet_img.shape[0]

        # 1. Encoder stage - extract multi-scale features
        pet_features = self.pet_encoder(pet_img)  # List of [B, C_i, H_i, W_i]
        ct_features = self.ct_encoder(ct_img)    # List of [B, C_i, H_i, W_i]

        # Project features to consistent dimension
        pet_features_proj = [self.feature_projections[i](feat) for i, feat in enumerate(pet_features)]
        ct_features_proj = [self.feature_projections[i](feat) for i, feat in enumerate(ct_features)]

        # 2. Transformer feature interaction with learnable queries
        pet_query_features = []
        ct_query_features = []

        for i in range(len(pet_features_proj)):
            pet_queries = self.transformer_interaction(pet_features_proj[i], i, 'pet')
            ct_queries = self.transformer_interaction(ct_features_proj[i], i, 'ct')
            pet_query_features.append(pet_queries)
            ct_query_features.append(ct_queries)

        # 3. Cross-modal interaction
        pet_enhanced_features = []
        ct_enhanced_features = []

        for i in range(len(pet_features_proj)):
            # PET queries interact with CT features
            pet_enhanced = self.cross_modal_pet2ct(pet_query_features[i], ct_features_proj[i])
            # CT queries interact with PET features
            ct_enhanced = self.cross_modal_ct2pet(ct_query_features[i], pet_features_proj[i])

            # Add interaction features to original query features
            pet_enhanced_features.append(pet_query_features[i] + pet_enhanced)
            ct_enhanced_features.append(ct_query_features[i] + ct_enhanced)

        # 4. CAM generation for classification supervision
        pet_cam, pet_classification = self.pet_cam_generator(pet_enhanced_features[-1])  # Use highest scale
        ct_cam, ct_classification = self.ct_cam_generator(ct_enhanced_features[-1])

        # 5. Point extraction from CAM
        pet_points = self.point_extractor(pet_cam, pet_features_proj[-1].shape)
        ct_points = self.point_extractor(ct_cam, ct_features_proj[-1].shape)

        # 6. SAM for weakly supervised mask generation
        sam_masks = {}

        if boxes is not None:
            # Box prompt masks
            box_prompts = self.sam_prompt_encoder(boxes=boxes)
            box_masks_pet = self.sam_mask_decoder(pet_features_proj[-1], box_prompts)
            box_masks_ct = self.sam_mask_decoder(ct_features_proj[-1], box_prompts)
            sam_masks['box_pet'] = box_masks_pet
            sam_masks['box_ct'] = box_masks_ct

            # Point prompt masks
            point_prompts_pet = self.sam_prompt_encoder(points=pet_points)
            point_prompts_ct = self.sam_prompt_encoder(points=ct_points)
            point_masks_pet = self.sam_mask_decoder(pet_features_proj[-1], point_prompts_pet)
            point_masks_ct = self.sam_mask_decoder(ct_features_proj[-1], point_prompts_ct)
            sam_masks['point_pet'] = point_masks_pet
            sam_masks['point_ct'] = point_masks_ct

        # 7. Final segmentation - add PET features to CT features
        fused_features = []
        for i in range(len(pet_features_proj)):
            # Convert query features back to spatial features
            B, N, C = pet_enhanced_features[i].shape
            H, W = pet_features_proj[i].shape[2], pet_features_proj[i].shape[3]

            # Simple spatial redistribution (can be improved with learned spatial mapping)
            queries_per_dim = int(math.sqrt(N))
            pet_spatial = pet_enhanced_features[i].mean(dim=1, keepdim=True).repeat(1, C, 1).view(B, C, queries_per_dim, queries_per_dim)
            pet_spatial = F.interpolate(pet_spatial, size=(H, W), mode='bilinear', align_corners=False)

            # Add PET features to CT features
            fused = ct_features_proj[i] + pet_spatial
            fused_features.append(fused)

        # Final decoder
        final_mask = self.final_decoder(fused_features)

        # 8. Loss computation
        losses = {}

        # Classification losses
        if gt_masks is not None:
            # Create tumor labels from masks
            tumor_labels = (gt_masks.sum(dim=[1, 2, 3]) > 0).long()
            losses['pet_classification'] = self.classification_loss(pet_classification, tumor_labels)
            losses['ct_classification'] = self.classification_loss(ct_classification, tumor_labels)

        # Weakly supervised losses
        if boxes is not None and sam_masks:
            # Use CT masks for primary supervision as CT has better spatial resolution
            losses['weakly_supervised'] = self.weakly_supervised_loss(
                final_mask, sam_masks['box_ct'], sam_masks['point_ct']
            )

        # Final segmentation loss
        if gt_masks is not None:
            losses['segmentation'] = self.segmentation_loss(final_mask, gt_masks.float())

        # Prepare outputs
        outputs = {
            'final_mask': torch.sigmoid(final_mask),
            'pet_classification': pet_classification,
            'ct_classification': ct_classification,
            'pet_cam': pet_cam,
            'ct_cam': ct_cam,
            'sam_masks': sam_masks,
            'losses': losses
        }

        return outputs


def build_model(config: dict = None):
    """
    Build the complete weakly supervised PET-CT model
    """
    if config is None:
        config = {
            'encoder_dims': [96, 192, 384, 768],
            'transformer_dim': 256,
            'num_queries': 100,
            'num_classes': 2,
            'sam_feature_dim': 256
        }

    model = NewWeaklySupervised_PETCT_Model(**config)
    return model


# ==================== Training Code ====================

import os
import time
import datetime
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import argparse
from easydict import EasyDict as edict


class WeaklySupervised_PETCT_Dataset(Dataset):
    """
    Dataset for weakly supervised PET-CT tumor segmentation
    Supports both bounding box annotations and full masks
    """
    def __init__(self, image_list, img_root, transforms=None, weak_supervision=True):
        super().__init__()
        self.image_list = image_list
        self.img_root = img_root
        self.transforms = transforms
        self.weak_supervision = weak_supervision

        # File suffixes
        self.pet_suffix = "_PET.png"
        self.ct_suffix = "_CT.png"
        self.mask_suffix = "_mask.png"
        self.box_suffix = "_box.txt"  # Bounding box annotations

    def _read_data(self, image_id):
        """Read PET, CT images and annotations"""
        pet_path = os.path.join(self.img_root, f"{image_id.split('_')[0]}/{image_id}{self.pet_suffix}")
        ct_path = os.path.join(self.img_root, f"{image_id.split('_')[0]}/{image_id}{self.ct_suffix}")
        mask_path = os.path.join(self.img_root, f"{image_id.split('_')[0]}/{image_id}{self.mask_suffix}")
        box_path = os.path.join(self.img_root, f"{image_id.split('_')[0]}/{image_id}{self.box_suffix}")

        # Read images
        pet = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

        # Read bounding box if available
        box = None
        if os.path.exists(box_path):
            with open(box_path, 'r') as f:
                box_line = f.readline().strip()
                if box_line:
                    # Format: x1,y1,x2,y2
                    box = [float(x) for x in box_line.split(',')]

        assert pet is not None, f"PET image not found: {pet_path}"
        assert ct is not None, f"CT image not found: {ct_path}"

        return pet, ct, mask, box

    def __getitem__(self, index):
        image_id = self.image_list[index]
        pet, ct, mask, box = self._read_data(image_id)

        # Normalize images
        pet = pet.astype(np.float32) / 255.0
        ct = ct.astype(np.float32) / 255.0

        # Convert to tensors
        pet = torch.from_numpy(pet).unsqueeze(0)  # [1, H, W]
        ct = torch.from_numpy(ct).unsqueeze(0)    # [1, H, W]

        sample = {
            'pet': pet,
            'ct': ct,
            'image_id': image_id
        }

        if mask is not None:
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
            sample['mask'] = mask

        if box is not None:
            sample['box'] = torch.tensor(box, dtype=torch.float32)

        return sample

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        pets = torch.stack([item['pet'] for item in batch])
        cts = torch.stack([item['ct'] for item in batch])

        batch_data = {
            'pet': pets,
            'ct': cts,
            'image_ids': [item['image_id'] for item in batch]
        }

        # Handle masks
        if 'mask' in batch[0]:
            masks = torch.stack([item['mask'] for item in batch])
            batch_data['masks'] = masks

        # Handle boxes
        if 'box' in batch[0]:
            boxes = torch.stack([item['box'] for item in batch])
            batch_data['boxes'] = boxes

        return batch_data


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (pred_mask > threshold).float()
    gt_binary = gt_mask.float()

    # IoU
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = intersection / (union + 1e-8)

    # Dice
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)

    # Accuracy
    correct = (pred_binary == gt_binary).sum()
    total = gt_binary.numel()
    accuracy = correct / total

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item()
    }


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, batch_data in enumerate(data_loader):
        pet = batch_data['pet'].to(device)
        ct = batch_data['ct'].to(device)

        boxes = batch_data.get('boxes', None)
        if boxes is not None:
            boxes = boxes.to(device)

        masks = batch_data.get('masks', None)
        if masks is not None:
            masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(pet, ct, boxes=boxes, gt_masks=masks)

        # Calculate total loss
        losses = outputs['losses']
        total_batch_loss = sum(losses.values())

        # Backward pass
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item()

        if batch_idx % print_freq == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{num_batches}], '
                  f'Loss: {total_batch_loss.item():.4f}')
            for loss_name, loss_value in losses.items():
                print(f'  {loss_name}: {loss_value.item():.4f}')

    return total_loss / num_batches


def evaluate(model, data_loader, device):
    """Evaluate the model"""
    model.eval()
    total_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}
    num_samples = 0

    with torch.no_grad():
        for batch_data in data_loader:
            pet = batch_data['pet'].to(device)
            ct = batch_data['ct'].to(device)

            if 'masks' not in batch_data:
                continue

            masks = batch_data['masks'].to(device)

            # Forward pass
            outputs = model(pet, ct)
            pred_masks = outputs['final_mask']

            # Calculate metrics
            for i in range(pred_masks.shape[0]):
                metrics = calculate_metrics(pred_masks[i], masks[i])
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_samples += 1

    # Average metrics
    if num_samples > 0:
        for key in total_metrics:
            total_metrics[key] /= num_samples

    return total_metrics


def prepare_dataset(data_root, split_ratio=0.8):
    """Prepare train/validation datasets"""
    # This is a placeholder - you should implement based on your data structure
    # For now, assuming you have train.txt and test.txt files
    train_file = os.path.join(data_root, 'train.txt')
    test_file = os.path.join(data_root, 'test.txt')

    if os.path.exists(train_file) and os.path.exists(test_file):
        with open(train_file, 'r') as f:
            train_list = [x.strip() for x in f.readlines()]
        with open(test_file, 'r') as f:
            test_list = [x.strip() for x in f.readlines()]
    else:
        # If split files don't exist, create them from all available data
        print("Split files not found, creating from available data...")
        # This would need to be implemented based on your data structure
        train_list = []
        test_list = []

    return train_list, test_list


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Weakly Supervised PET-CT Tumor Segmentation')
    parser.add_argument('--data_root', type=str, default='./data/PCLT20k/',
                       help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=6e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Prepare datasets
    train_list, test_list = prepare_dataset(args.data_root)
    print(f'Train samples: {len(train_list)}, Test samples: {len(test_list)}')

    train_dataset = WeaklySupervised_PETCT_Dataset(train_list, args.data_root, transforms=True)
    val_dataset = WeaklySupervised_PETCT_Dataset(test_list, args.data_root, transforms=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=WeaklySupervised_PETCT_Dataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=WeaklySupervised_PETCT_Dataset.collate_fn,
        pin_memory=True
    )

    # Create model
    model = build_model()
    model.to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params:,}')

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_dice = checkpoint['best_dice']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Training loop
    print("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.print_freq)
        scheduler.step()

        print(f'Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0:
            metrics = evaluate(model, val_loader, device)
            print(f'Validation - IoU: {metrics["iou"]:.4f}, Dice: {metrics["dice"]:.4f}, Acc: {metrics["accuracy"]:.4f}')

            # Save best model
            if metrics['dice'] > best_dice:
                best_dice = metrics['dice']
                save_path = os.path.join(args.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_dice': best_dice,
                    'metrics': metrics
                }, save_path)
                print(f'New best model saved with Dice: {best_dice:.4f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice
            }, save_path)

    # Final evaluation
    final_metrics = evaluate(model, val_loader, device)
    print(f'Final Results - IoU: {final_metrics["iou"]:.4f}, Dice: {final_metrics["dice"]:.4f}, Acc: {final_metrics["accuracy"]:.4f}')

    total_time = time.time() - start_time
    print(f'Training completed in {total_time/3600:.2f} hours')
    print(f'Best Dice score: {best_dice:.4f}')


if __name__ == "__main__":
    # Check if running training
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        main()
    else:
        # Test the model
        print("Testing model...")
        model = build_model()

        # Create dummy inputs
        batch_size = 2
        pet_img = torch.randn(batch_size, 1, 512, 512)
        ct_img = torch.randn(batch_size, 1, 512, 512)
        boxes = torch.tensor([[100, 100, 200, 200], [150, 150, 250, 250]], dtype=torch.float32)
        gt_masks = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()

        # Forward pass
        with torch.no_grad():
            outputs = model(pet_img, ct_img, boxes, gt_masks)

        print("Model built successfully!")
        print(f"Final mask shape: {outputs['final_mask'].shape}")
        print(f"PET classification shape: {outputs['pet_classification'].shape}")
        print(f"CT classification shape: {outputs['ct_classification'].shape}")
        if outputs['losses']:
            print(f"Losses: {list(outputs['losses'].keys())}")

        print("\nTo start training, run:")
        print("python new_model.py train --data_root /path/to/your/data --batch_size 4 --epochs 50")