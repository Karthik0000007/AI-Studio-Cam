"""COCO Multi-Label CNN Training - Simplified"""

import os, torch, optuna, json, argparse, logging
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None, max_samples=None):
        self.root_dir, self.transform = root_dir, transform
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create mappings
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.class_names = list(self.categories.values())
        image_info = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        image_labels = defaultdict(set)
        for ann in coco_data['annotations']:
            if ann['category_id'] in self.categories:
                image_labels[ann['image_id']].add(ann['category_id'])
        
        # Create samples with multi-hot labels
        self.samples = []
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        for image_id, category_ids in image_labels.items():
            if image_id in image_info:
                image_path = os.path.join(root_dir, image_info[image_id])
                if os.path.exists(image_path):
                    label_vector = torch.zeros(len(self.class_names))
                    for cat_id in category_ids:
                        class_idx = class_to_idx[self.categories[cat_id]]
                        label_vector[class_idx] = 1.0
                    self.samples.append((image_path, label_vector))
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        logger.info(f"Loaded {len(self.samples)} images with {len(self.class_names)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image) if self.transform else image, label
        except:
            # Return black image on failure
            black_img = torch.zeros(3, 224, 224)
            return black_img, label


class CustomCNN(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        # Fixed 3-layer CNN (32→64→128 filters)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Additional fully connected layer (128→64→num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)


def create_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced multi-label data"""
    # Calculate label frequencies
    label_counts = torch.zeros(len(dataset.class_names))
    for _, labels in dataset.samples:
        label_counts += labels
    
    label_weights = 1.0 / (label_counts + 1e-6) 
    
    sample_weights = []
    for _, labels in dataset.samples:
        active_labels = labels > 0
        if active_labels.sum() > 0:
            weight = (label_weights * active_labels).sum() / active_labels.sum()
        else:
            weight = 1.0
        sample_weights.append(weight.item())
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def create_datasets(dataset_path, max_samples=None):
    """Create train/val/test datasets from COCO"""
    train_dataset = COCODataset(os.path.join(dataset_path, 'train2017'),
                               os.path.join(dataset_path, 'annotations', 'instances_train2017.json'),
                               get_transforms(train=True), max_samples)
    val_full = COCODataset(os.path.join(dataset_path, 'val2017'),
                          os.path.join(dataset_path, 'annotations', 'instances_val2017.json'),
                          get_transforms(train=False), max_samples // 4 if max_samples else None)
    
    val_size = len(val_full) // 2
    val_dataset, test_dataset = random_split(val_full, [val_size, len(val_full) - val_size])
    return train_dataset, val_dataset, test_dataset


def run_epoch(model, loader, criterion, optimizer=None, accumulation_steps=1):
    """Enhanced training epoch with gradient accumulation and real-time metrics"""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(is_training):
        pbar = tqdm(loader, desc=f"{'Train' if is_training else 'Val'}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            output = model(data)
            loss = criterion(output, target)
            
            # Scale loss for gradient accumulation
            if is_training:
                loss = loss / accumulation_steps
                loss.backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Metrics calculation
            total_loss += loss.item() * accumulation_steps
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == target).float().sum().item()
            total += target.numel()
            
            # Enhanced progress bars with real-time metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    return total_loss / len(loader), 100 * correct / total


def train_model(params, train_dataset, val_dataset, num_classes, max_epochs, patience, trial=None):
    """Train model with given parameters using weighted sampling and gradient accumulation"""
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders with weighted sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        sampler=train_sampler,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=device.type == 'cuda'
    )
    
    model = CustomCNN(num_classes, params['dropout_rate']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params.get('weight_decay', 1e-4))
    
    # Calculate gradient accumulation steps for effective batch size of 64
    accumulation_steps = max(1, 64 // params['batch_size'])
    
    best_val_loss, best_model_state, patience_counter = float('inf'), None, 0
    
    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{max_epochs}")
        
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, accumulation_steps)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        
        logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss, best_model_state, patience_counter = val_loss, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return val_acc, best_model_state


def optimize_hyperparameters(train_dataset, val_dataset, num_classes, n_trials=15):
    """More sophisticated Optuna setup with pruning"""

    def objective(trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        
        # Quick training for optimization (fewer epochs)
        val_acc, _ = train_model(params, train_dataset, val_dataset, num_classes, 8, 3, trial)
        return val_acc
    
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
    
    # More sophisticated Optuna setup with pruning
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 min timeout
    
    logger.info(f"Best validation accuracy: {study.best_value:.2f}%")
    logger.info(f"Best hyperparameters: {study.best_params}")
    
    return study.best_params


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='COCO Multi-Label CNN Training')
    parser.add_argument('--dataset_path', default='./data/coco_dataset', help='Path to COCO dataset')
    parser.add_argument('--models_dir', default='models', help='Directory to save models')
    parser.add_argument('--trials', type=int, default=15, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit number of samples for testing')
    
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    
    required_files = [
        os.path.join(args.dataset_path, 'annotations', 'instances_train2017.json'),
        os.path.join(args.dataset_path, 'annotations', 'instances_val2017.json')
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            logger.info("Please ensure the COCO dataset is properly structured")
            return
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = create_datasets(args.dataset_path, args.max_samples)
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return
    
    num_classes = len(train_dataset.class_names)
    logger.info(f"Loaded {num_classes} classes, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimize hyperparameters
    best_params = optimize_hyperparameters(train_dataset, val_dataset, num_classes, args.trials)
    
    # Final training
    logger.info("Final training with best hyperparameters...")
    logger.info(f"Parameters: {best_params}")
    
    best_val_acc, best_model_state = train_model(best_params, train_dataset, val_dataset, num_classes, args.epochs, 10)
    
    # Test evaluation
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    model = CustomCNN(num_classes, best_params['dropout_rate']).to(device)
    model.load_state_dict(best_model_state)
    _, test_acc = run_epoch(model, test_loader, nn.BCEWithLogitsLoss())
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.models_dir, f'coco_multilabel_cnn_{timestamp}.pth')
    
    torch.save({
        'model_state_dict': best_model_state,
        'num_classes': num_classes,
        'class_names': train_dataset.class_names,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'hyperparameters': best_params
    }, model_path)
    
    # Calculate model size
    model = CustomCNN(num_classes, best_params['dropout_rate'])
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Training completed! Val: {best_val_acc:.2f}%, Test: {test_acc:.2f}%")
    logger.info(f"Model Parameters: {total_params:,}")
    logger.info(f"Model Size: {total_params * 4 / 1024 / 1024:.1f} MB")
    logger.info(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
