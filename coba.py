# -*- coding: utf-8 -*-
"""Improved SSD PyTorch ResNet34 with mAP@0.5 Evaluation"""

import torch
import torchvision
from torchvision.models import resnet34, resnet50, resnet101
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2
from PIL import Image
from tqdm import tqdm
import itertools
import math
import os
from collections import defaultdict

# Download dataset
# !curl -L "https://app.roboflow.com/ds/pEAhd79vNm?key=BmEzXTcsiI" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

class OrchidDiseaseDataset(Dataset):
    def __init__(self, coco_annotation, img_dir, transform=None):
        self.coco = COCO(coco_annotation)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())
        self.categories = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}

        # Validate all image paths during initialization
        self.valid_ids = []
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            if os.path.exists(img_path):
                self.valid_ids.append(img_id)
            else:
                print(f"Warning: Image not found - {img_path}")

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Read image with multiple fallback methods
        try:
            # First try with OpenCV
            img = cv2.imread(img_path)
            if img is None:
                # Fallback to PIL if OpenCV fails
                img = Image.open(img_path)
                img = np.array(img)
                if len(img.shape) == 2:  # Convert grayscale to RGB
                    img = np.stack((img,)*3, axis=-1)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")

        # Get original dimensions for normalization
        orig_h, orig_w = img.shape[:2]

        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']
            # Convert from [x,y,w,h] to [x1,y1,x2,y2] and normalize
            x1 = max(0, bbox[0] / orig_w)
            y1 = max(0, bbox[1] / orig_h)
            x2 = min(1, (bbox[0] + bbox[2]) / orig_w)
            y2 = min(1, (bbox[1] + bbox[3]) / orig_h)

            # Only add valid boxes
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            # Return a dummy box if no valid boxes found
            boxes.append([0, 0, 0.1, 0.1])
            labels.append(1)  # Use class 1 instead of 0 (background)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.valid_ids)

# Function to calculate IoU
def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_ap_per_class(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, class_id, iou_threshold=0.5):
    """Calculate Average Precision for a single class"""
    
    # Filter predictions and ground truth for this class
    class_pred_mask = pred_labels == class_id
    class_true_mask = true_labels == class_id
    
    if not np.any(class_pred_mask) and not np.any(class_true_mask):
        return None  # No predictions or ground truth for this class
    
    if not np.any(class_true_mask):
        return 0.0  # No ground truth for this class
    
    if not np.any(class_pred_mask):
        return 0.0  # No predictions for this class
    
    class_pred_boxes = pred_boxes[class_pred_mask]
    class_pred_scores = pred_scores[class_pred_mask]
    class_true_boxes = true_boxes[class_true_mask]
    
    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(class_pred_scores)[::-1]
    class_pred_boxes = class_pred_boxes[sorted_indices]
    class_pred_scores = class_pred_scores[sorted_indices]
    
    num_gt = len(class_true_boxes)
    tp = np.zeros(len(class_pred_boxes))
    fp = np.zeros(len(class_pred_boxes))
    
    # Track which ground truth boxes have been matched
    gt_matched = np.zeros(num_gt, dtype=bool)
    
    for i, pred_box in enumerate(class_pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(class_true_boxes):
            if gt_matched[j]:
                continue
                
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Calculate AP using the 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap

def calculate_map50(pred_boxes_list, pred_labels_list, pred_scores_list, 
                   true_boxes_list, true_labels_list, num_classes, iou_threshold=0.5):
    """Calculate mAP@0.5 across all images and classes"""
    
    # Concatenate all predictions and ground truth
    all_pred_boxes = np.concatenate(pred_boxes_list) if pred_boxes_list else np.array([]).reshape(0, 4)
    all_pred_labels = np.concatenate(pred_labels_list) if pred_labels_list else np.array([])
    all_pred_scores = np.concatenate(pred_scores_list) if pred_scores_list else np.array([])
    all_true_boxes = np.concatenate(true_boxes_list) if true_boxes_list else np.array([]).reshape(0, 4)
    all_true_labels = np.concatenate(true_labels_list) if true_labels_list else np.array([])
    
    if len(all_pred_boxes) == 0 or len(all_true_boxes) == 0:
        return 0.0, {}
    
    # Calculate AP for each class
    class_aps = {}
    valid_aps = []
    
    for class_id in range(1, num_classes):  # Skip background class (0)
        ap = calculate_ap_per_class(
            all_pred_boxes, all_pred_labels, all_pred_scores,
            all_true_boxes, all_true_labels, class_id, iou_threshold
        )
        
        if ap is not None:
            class_aps[class_id] = ap
            valid_aps.append(ap)
    
    # Calculate mean AP
    map50 = np.mean(valid_aps) if valid_aps else 0.0
    
    return map50, class_aps

def calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score"""
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0

    tp = 0
    matched_true = set()

    # Sort predictions by confidence score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]

    for i, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
        best_iou = 0
        best_idx = -1

        for j, (tb, tl) in enumerate(zip(true_boxes, true_labels)):
            if j in matched_true:
                continue
            if pl == tl:
                iou = calculate_iou(pb, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_true.add(best_idx)

    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

class CustomSSDBackbone(torch.nn.Module):
    """Custom SSD backbone with proper feature extraction"""
    def __init__(self, backbone_name='resnet34'):
        super().__init__()
        
        if backbone_name == 'resnet34':
            backbone = resnet34(weights='IMAGENET1K_V1')
            self.out_channels = 512
        elif backbone_name == 'resnet50':
            backbone = resnet50(weights='IMAGENET1K_V1')
            self.out_channels = 2048
        elif backbone_name == 'resnet101':
            backbone = resnet101(weights='IMAGENET1K_V1')
            self.out_channels = 2048
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Extract feature layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Additional layers for multi-scale features
        self.extra_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(self.out_channels, 512, kernel_size=3, padding=1),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        ])
        
    def forward(self, x):
        features = []
        
        # Base ResNet forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        features.append(x)
        
        # Additional feature maps
        for layer in self.extra_layers:
            x = torch.nn.functional.relu(layer(x))
            features.append(x)
            
        return features

def build_ssd_model(backbone_name='resnet34', num_classes=4):
    """Build proper SSD model with custom backbone"""
    
    # Create custom backbone
    backbone = CustomSSDBackbone(backbone_name)
    
    # Use SSD300 as template
    model = ssd300_vgg16(weights=None, num_classes=num_classes)
    
    # Replace backbone
    model.backbone = backbone
    
    # Update head input channels to match our backbone
    if backbone_name == 'resnet34':
        in_channels = [512, 512, 512, 256, 256, 256]
    else:  # resnet50/101
        in_channels = [2048, 512, 512, 256, 256, 256]
    
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Recreate classification and regression heads
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    model.head.regression_head = SSDRegressionHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
    )
    
    return model

def evaluate_model(model, val_loader, device, num_classes, confidence_threshold=0.5):
    """Comprehensive model evaluation with mAP@0.5"""
    model.eval()
    
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_true_boxes = []
    all_true_labels = []
    
    precisions = []
    recalls = []
    f1_scores = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                # Process predictions
                if len(pred['boxes']) > 0:
                    # Filter by confidence
                    high_conf_mask = pred['scores'] >= confidence_threshold
                    pred_boxes = pred['boxes'][high_conf_mask].cpu().numpy()
                    pred_labels = pred['labels'][high_conf_mask].cpu().numpy()
                    pred_scores = pred['scores'][high_conf_mask].cpu().numpy()
                else:
                    pred_boxes = np.array([]).reshape(0, 4)
                    pred_labels = np.array([])
                    pred_scores = np.array([])
                
                # Process ground truth
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                
                # Store for mAP calculation
                if len(pred_boxes) > 0:
                    all_pred_boxes.append(pred_boxes)
                    all_pred_labels.append(pred_labels)
                    all_pred_scores.append(pred_scores)
                
                if len(true_boxes) > 0:
                    all_true_boxes.append(true_boxes)
                    all_true_labels.append(true_labels)
                
                # Calculate per-image metrics
                if len(pred_boxes) == 0:
                    precisions.append(0.0 if len(true_boxes) > 0 else 1.0)
                    recalls.append(0.0)
                    f1_scores.append(0.0)
                else:
                    p, r, f1 = calculate_metrics(
                        pred_boxes, pred_labels, pred_scores,
                        true_boxes, true_labels
                    )
                    precisions.append(p)
                    recalls.append(r)
                    f1_scores.append(f1)
    
    # Calculate mAP@0.5
    map50, class_aps = calculate_map50(
        all_pred_boxes, all_pred_labels, all_pred_scores,
        all_true_boxes, all_true_labels, num_classes
    )
    
    # Calculate average metrics
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'map50': map50,
        'class_aps': class_aps
    }

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.005, num_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    val_losses = []
    val_metrics = {
        'precision': [], 
        'recall': [], 
        'f1': [],
        'map50': [],
        'class_aps': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        train_batch_count = 0

        for images, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            train_batch_count += 1

        train_losses.append(epoch_loss / train_batch_count)
        lr_scheduler.step()

        # Validation phase - Calculate loss
        model.train()  # Need train mode for loss calculation
        val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                batch_loss = sum(loss for loss in loss_dict.values())
                val_loss += batch_loss.item()
                val_batch_count += 1

        val_losses.append(val_loss / max(1, val_batch_count))

        # Comprehensive evaluation
        metrics = evaluate_model(model, val_loader, device, num_classes)
        
        val_metrics['precision'].append(metrics['precision'])
        val_metrics['recall'].append(metrics['recall'])
        val_metrics['f1'].append(metrics['f1'])
        val_metrics['map50'].append(metrics['map50'])
        val_metrics['class_aps'].append(metrics['class_aps'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_losses[-1]:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f}")
        
        # Print per-class AP
        if metrics['class_aps']:
            print("Per-class AP:")
            for class_id, ap in metrics['class_aps'].items():
                print(f"  Class {class_id}: {ap:.4f}")

    return model, train_losses, val_losses, val_metrics

def plot_training_curves(train_losses, val_losses, val_metrics):
    """Plot comprehensive training curves including mAP@0.5"""
    plt.figure(figsize=(20, 10))

    # Plot 1: Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Precision and Recall
    plt.subplot(2, 3, 2)
    if val_metrics['precision'] and val_metrics['recall']:
        plt.plot(range(1, len(val_metrics['precision']) + 1), val_metrics['precision'],
                label='Precision', marker='o')
        plt.plot(range(1, len(val_metrics['recall']) + 1), val_metrics['recall'],
                label='Recall', marker='s')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: F1 Score
    plt.subplot(2, 3, 3)
    if val_metrics['f1']:
        plt.plot(range(1, len(val_metrics['f1']) + 1), val_metrics['f1'],
                label='F1 Score', marker='o', color='green')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: mAP@0.5
    plt.subplot(2, 3, 4)
    if val_metrics['map50']:
        plt.plot(range(1, len(val_metrics['map50']) + 1), val_metrics['map50'],
                label='mAP@0.5', marker='d', color='red', linewidth=2)
        plt.title('Mean Average Precision @ IoU=0.5')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 5: All metrics together
    plt.subplot(2, 3, 5)
    if val_metrics['precision'] and val_metrics['recall'] and val_metrics['f1'] and val_metrics['map50']:
        epochs = range(1, len(val_metrics['precision']) + 1)
        plt.plot(epochs, val_metrics['precision'], label='Precision', marker='o')
        plt.plot(epochs, val_metrics['recall'], label='Recall', marker='s')
        plt.plot(epochs, val_metrics['f1'], label='F1 Score', marker='^')
        plt.plot(epochs, val_metrics['map50'], label='mAP@0.5', marker='d', linewidth=2)
        plt.title('All Metrics Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 6: Per-class AP for last epoch
    plt.subplot(2, 3, 6)
    if val_metrics['class_aps'] and val_metrics['class_aps'][-1]:
        class_ids = list(val_metrics['class_aps'][-1].keys())
        aps = list(val_metrics['class_aps'][-1].values())
        plt.bar(class_ids, aps, color='skyblue', alpha=0.7)
        plt.title('Per-Class AP (Final Epoch)')
        plt.xlabel('Class ID')
        plt.ylabel('Average Precision')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (class_id, ap) in enumerate(zip(class_ids, aps)):
            plt.text(class_id, ap + 0.01, f'{ap:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Initialize datasets
    train_dataset = OrchidDiseaseDataset(
        coco_annotation='/kaggle/working/train/_annotations.coco.json',
        img_dir='/kaggle/working/train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    val_dataset = OrchidDiseaseDataset(
        coco_annotation='/kaggle/working/valid/_annotations.coco.json',
        img_dir='/kaggle/working/valid',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    # Build model
    num_classes = 4  # 3 classes + background
    model = build_ssd_model(backbone_name='resnet34', num_classes=num_classes)

    # Train model with comprehensive evaluation
    trained_model, train_losses, val_losses, val_metrics = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=0.001,
        num_classes=num_classes
    )

    # Visualize results
    plot_training_curves(train_losses, val_losses, val_metrics)

    # Save model
    torch.save(trained_model.state_dict(), 'ssd_orchid_model_improved.pth')
    
    # Print final metrics summary
    print("\n" + "="*50)
    print("FINAL TRAINING SUMMARY")
    print("="*50)
    if val_metrics['map50']:
        print(f"Best mAP@0.5: {max(val_metrics['map50']):.4f}")
        print(f"Final mAP@0.5: {val_metrics['map50'][-1]:.4f}")
    if val_metrics['f1']:
        print(f"Best F1 Score: {max(val_metrics['f1']):.4f}")
        print(f"Final F1 Score: {val_metrics['f1'][-1]:.4f}")
    if val_metrics['precision']:
        print(f"Final Precision: {val_metrics['precision'][-1]:.4f}")
    if val_metrics['recall']:
        print(f"Final Recall: {val_metrics['recall'][-1]:.4f}")