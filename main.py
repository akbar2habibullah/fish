import os
import json
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from tqdm import tqdm
import torch.nn as nn

# ---------------------
# Custom Dataset Classes
# ---------------------

class FishSeedDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        Args:
            root (str): Root directory containing 'images/' and 'annotations.json'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transforms = transforms

        # Load annotation file
        annotation_file = os.path.join(root, "annotations.json")
        with open(annotation_file) as f:
            self.coco = json.load(f)

        # Create mappings
        self.image_id_map = {img['file_name']: img['id'] for img in self.coco['images']}
        self.annotations = defaultdict(list)
        for ann in self.coco['annotations']:
            # Associate annotations with file names
            image_info = next((img for img in self.coco['images'] if img['id'] == ann['image_id']), None)
            if image_info:
                self.annotations[image_info['file_name']].append(ann)

        # List of image filenames
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        ann = self.annotations[self.imgs[idx]]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in ann:
            xmin, ymin, width, height = obj['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(1)  # 'fish_seed' category
            areas.append(obj['area'])
            iscrowd.append(obj['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([self.coco['images'][idx]['id']])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target

class FishSeedCountingDataset(Dataset):
    def __init__(self, root, counts_file, transforms=None):
        """
        Args:
            root (str): Root directory containing 'images/'.
            counts_file (str): Path to 'counts.csv'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transforms = transforms
        self.counts_df = pd.read_csv(counts_file)
        self.imgs = list(self.counts_df['filename'])
        self.counts = list(self.counts_df['count'])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        count = self.counts[idx]

        if self.transforms:
            img = self.transforms(img)

        return img, count

# ---------------------
# Combined Dataset
# ---------------------

class CombinedDataset(Dataset):
    def __init__(self, phase1_dataset, phase2_dataset):
        """
        Args:
            phase1_dataset (Dataset): Dataset for Phase 1 (Detection).
            phase2_dataset (Dataset): Dataset for Phase 2 (Counting).
        """
        self.phase1_dataset = phase1_dataset
        self.phase2_dataset = phase2_dataset

    def __len__(self):
        return len(self.phase1_dataset) + len(self.phase2_dataset)

    def __getitem__(self, idx):
        if idx < len(self.phase1_dataset):
            img, target = self.phase1_dataset[idx]
            return img, target, None  # No count label
        else:
            img, count = self.phase2_dataset[idx - len(self.phase1_dataset)]
            return img, None, count  # No target annotations

# ---------------------
# Model Definition
# ---------------------

class FasterRCNNWithCount(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNNWithCount, self).__init__()
        # Initialize Faster R-CNN
        self.faster_rcnn = get_model(num_classes)
        self.num_classes = num_classes

        # Add a regression head for counting
        # We'll use average pooling on the backbone features and then a fully connected layer
        # Alternatively, more sophisticated feature aggregation can be used
        # Adjust the input features based on the backbone
        self.count_regressor = nn.Sequential(
            nn.Linear(2048, 512),  # Assuming ResNet-50 backbone output channels
            nn.ReLU(),
            nn.Linear(512, 1)  # Predicting the count
        )

    def forward(self, images, targets=None, counts=None):
        """
        Args:
            images: List of images tensors.
            targets: List of target dictionaries (for detection).
            counts: List or tensor of counts (for counting).
        Returns:
            If training:
                Combined loss dictionary
            Else:
                Detections
        """
        if self.training:
            # Detection losses
            loss_dict = self.faster_rcnn(images, targets)

            # Counting losses
            if counts is not None:
                # Extract features from the backbone
                # For Faster R-CNN, the backbone is a ResNet-50 by default
                # We'll use the output of the backbone before the RPN
                # Access backbone features
                features = self.faster_rcnn.backbone(images.tensors)
                # Typically, for ResNet backbones, features['0'] is the top layer
                backbone_features = features['0']  # Shape: [batch_size, 2048, H, W]

                # Global Average Pooling
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(backbone_features, (1, 1))
                pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [batch_size, 2048]

                # Regression Head to predict counts
                count_preds = self.count_regressor(pooled_features).squeeze(1)  # [batch_size]

                # Convert counts to tensor
                count_targets = torch.tensor(counts, dtype=torch.float32).to(images.tensors.device)

                # Compute L1 Loss for counting
                count_loss = nn.L1Loss()(count_preds, count_targets)

                # Combine losses
                total_loss = sum(loss for loss in loss_dict.values()) + count_loss
                loss_dict['loss_count'] = count_loss
                loss_dict['total_loss'] = total_loss

                return loss_dict
            else:
                return self.faster_rcnn(images, targets)
        else:
            # Inference
            detections = self.faster_rcnn(images)
            return detections

def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def train_phase1(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch} Phase 1 Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Total loss is the sum of individual losses
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} Phase 1 - Average Loss: {avg_loss:.4f}")

def evaluate_phase1(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Phase 1 Evaluation"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)

            # Total loss
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Phase 1 Evaluation - Average Loss: {avg_loss:.4f}")
    return avg_loss

def train_phase2(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for images, targets, counts in tqdm(data_loader, desc=f"Epoch {epoch} Phase 2 Training"):
        images = list(image.to(device) for image in images)
        batch_targets = []
        batch_counts = []
        for tgt, cnt in zip(targets, counts):
            if tgt is not None:
                batch_targets.append(tgt)
            if cnt is not None:
                batch_counts.append(cnt)

        if len(batch_targets) > 0:
            # Forward pass with targets and counts
            loss_dict = model(images, targets=batch_targets, counts=batch_counts)
            losses = loss_dict['total_loss']

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        else:
            continue  # Skip if no targets and counts

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} Phase 2 - Average Loss: {avg_loss:.4f}")

def evaluate_counting(model, data_loader, device, threshold=0.5):
    model.eval()
    predicted_counts = []
    actual_counts = []
    with torch.no_grad():
        for images, counts in tqdm(data_loader, desc="Phase 2 Counting Evaluation"):
            images = list(image.to(device) for image in images)
            outputs = model.faster_rcnn(images)

            for output, count in zip(outputs, counts):
                detected = 0
                for score, label in zip(output['scores'], output['labels']):
                    if score > threshold and label == 1:
                        detected += 1
                predicted_counts.append(detected)
                actual_counts.append(count)

    predicted_counts = np.array(predicted_counts)
    actual_counts = np.array(actual_counts)

    mae = np.mean(np.abs(predicted_counts - actual_counts))
    rmse = np.sqrt(np.mean((predicted_counts - actual_counts) ** 2))

    print(f"Phase 2 Counting - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

def visualize_predictions_phase1(model, dataset, device, threshold=0.5, num_samples=5):
    model.eval()
    for i in range(num_samples):
        img, target = dataset[i]
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        img_np = img.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img_np)
        true_count = len(target['boxes'])
        pred_count = 0
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score > threshold and label == 1:
                pred_count += 1
                xmin, ymin, xmax, ymax = box.cpu()
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

        plt.title(f"Ground Truth: {true_count}, Predicted: {pred_count}")
        plt.axis('off')
        plt.show()

def visualize_predictions_phase2(model, dataset, device, threshold=0.5, num_samples=5):
    model.faster_rcnn.eval()
    for i in range(num_samples):
        img, count = dataset[i]
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model.faster_rcnn(img_input)[0]

        img_np = img.permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(1)
        ax.imshow(img_np)
        pred_count = 0
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score > threshold and label == 1:
                pred_count += 1
                xmin, ymin, xmax, ymax = box.cpu()
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

        plt.title(f"Actual: {count}, Predicted: {pred_count}")
        plt.axis('off')
        plt.show()

def main():
    # Paths to dataset phases
    phase1_root = 'dataset/phase1'  # Replace with your actual path
    phase2_root = 'dataset/phase2'  # Replace with your actual path
    phase2_counts_file = os.path.join(phase2_root, "counts.csv")

    # Define transforms
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Create datasets
    dataset_phase1 = FishSeedDetectionDataset(phase1_root, transforms=transform)
    dataset_phase2 = FishSeedCountingDataset(phase2_root, phase2_counts_file, transforms=transform)

    # Split Phase 1 into training and validation
    val_split = 0.1
    num_val = int(len(dataset_phase1) * val_split)
    num_train = len(dataset_phase1) - num_val
    dataset_phase1_train, dataset_phase1_val = torch.utils.data.random_split(dataset_phase1, [num_train, num_val])

    # DataLoaders for Phase 1
    data_loader_phase1_train = DataLoader(dataset_phase1_train, batch_size=4, shuffle=True, num_workers=4,
                                         collate_fn=lambda x: tuple(zip(*x)))
    data_loader_phase1_val = DataLoader(dataset_phase1_val, batch_size=4, shuffle=False, num_workers=4,
                                       collate_fn=lambda x: tuple(zip(*x)))

    # DataLoader for Phase 2
    # For Phase 2, we need to provide images with counts. We'll create a combined dataset.
    # To handle both datasets, we'll use a custom collate_fn
    combined_dataset = CombinedDataset(dataset_phase1_train, dataset_phase2)
    data_loader_combined = DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=4,
                                      collate_fn=lambda x: tuple(zip(*x)))

    # Define the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Define the number of classes (background + fish_seed)
    num_classes = 2

    # Initialize the model
    model = FasterRCNNWithCount(num_classes)
    model.to(device)

    # Define optimizer (commonly SGD for Faster R-CNN)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs_phase1 = 5
    num_epochs_phase2 = 10

    # Phase 1: Train object detection
    print("Starting Phase 1 Training (Object Detection)...")
    for epoch in range(1, num_epochs_phase1 + 1):
        train_phase1(model.faster_rcnn, optimizer, data_loader_phase1_train, device, epoch)
        evaluate_phase1(model.faster_rcnn, data_loader_phase1_val, device)
        lr_scheduler.step()

    print("Phase 1 Training Completed.\n")

    # Save the Phase 1 model
    torch.save(model.state_dict(), "fasterrcnn_fish_seed_phase1.pth")
    print("Phase 1 Model saved as fasterrcnn_fish_seed_phase1.pth")

    # Visualization for Phase 1
    visualize_predictions_phase1(model.faster_rcnn, dataset_phase1_val, device, threshold=0.5, num_samples=3)

    # Phase 2: Fine-tune with counting
    print("Starting Phase 2 Training (Object Detection + Counting)...")
    for epoch in range(1, num_epochs_phase2 + 1):
        train_phase2(model, optimizer, data_loader_combined, device, epoch)
        lr_scheduler.step()

    print("Phase 2 Training Completed.\n")

    # Save the Phase 2 model
    torch.save(model.state_dict(), "fasterrcnn_fish_seed_phase2.pth")
    print("Phase 2 Model saved as fasterrcnn_fish_seed_phase2.pth")

    # Evaluation for Phase 2
    mae, rmse = evaluate_counting(model, DataLoader(dataset_phase2, batch_size=4, shuffle=False, num_workers=4), device, threshold=0.5)

    # Visualization for Phase 2
    visualize_predictions_phase2(model, dataset_phase2, device, threshold=0.5, num_samples=3)

    # Optionally, plot predicted vs actual counts
    predicted_counts = []
    actual_counts = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset_phase2)):
            img, count = dataset_phase2[i]
            img_input = img.unsqueeze(0).to(device)
            prediction = model.faster_rcnn(img_input)[0]
            pred_count = sum((score > 0.5 and label == 1) for score, label in zip(prediction['scores'], prediction['labels']))
            predicted_counts.append(pred_count)
            actual_counts.append(count)

    plt.figure(figsize=(8,6))
    plt.scatter(actual_counts, predicted_counts, alpha=0.6)
    plt.plot([actual_counts.min(), actual_counts.max()], [actual_counts.min(), actual_counts.max()], 'r--')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.title('Actual vs Predicted Fish Seed Counts')
    plt.show()

