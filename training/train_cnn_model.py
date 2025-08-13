"""Custom CNN Training with Hyperparameter Optimization"""

import os, torch, optuna
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = os.cpu_count()


class CustomDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), class_idx))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            # Open image and convert to RGBA first to handle transparency, then to RGB
            image = Image.open(img_path).convert('RGBA').convert('RGB')
            return self.transform(image) if self.transform else image, label
        except:
            black_img = Image.new('RGB', (224, 224), (0, 0, 0))
            return self.transform(black_img) if self.transform else black_img, label


class CustomCNN(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.2, num_filters=16):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(num_filters, num_filters * 2, 3, padding=1), nn.BatchNorm2d(num_filters * 2), nn.ReLU(True),
            nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1), nn.BatchNorm2d(num_filters * 2), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, padding=1), nn.BatchNorm2d(num_filters * 4), nn.ReLU(True),
            nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1), nn.BatchNorm2d(num_filters * 4), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(num_filters * 4, num_filters * 8, 3, padding=1), nn.BatchNorm2d(num_filters * 8), nn.ReLU(True),
            nn.Conv2d(num_filters * 8, num_filters * 8, 3, padding=1), nn.BatchNorm2d(num_filters * 8), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(num_filters * 8, num_filters * 4), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(num_filters * 4, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def create_datasets_with_transforms(dataset_path):
    """Create and split datasets with basic transforms (no augmentation)"""
    # Basic transform 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create base dataset and split
    dataset = CustomDataset(dataset_path, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset, dataset.classes


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    
    total_loss, correct, total = 0, 0, 0
    context_manager = torch.amp.autocast('cuda', dtype=torch.float16) if is_training else torch.no_grad()
    
    with context_manager:
        for data, target in loader:
            # Optimization: channels_last memory format for better GPU performance
            data = data.to(device, non_blocking=True, memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            
            if is_training:
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return total_loss / len(loader), 100 * correct / total


def train_with_params(max_epochs, patience_limit, params, train_loader, val_loader, num_classes, trial=None):
    model = CustomCNN(num_classes, params['dropout_rate'], params['num_filters']).to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler()
    
    best_val_acc, best_model_state, patience_counter, training_history = 0, None, 0, []
    
    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()
        
        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune(): raise optuna.TrialPruned()
        
        training_history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                               'val_loss': val_loss, 'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']})
        
        if val_acc > best_val_acc:
            best_val_acc, best_model_state, patience_counter = val_acc, model.state_dict().copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                break
    
    return best_val_acc, best_model_state, training_history


def objective(trial, train_dataset, val_dataset, num_classes):
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'num_filters': trial.suggest_categorical('num_filters', [16, 24, 32])
    }
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    val_acc, _, _ = train_with_params(8, 3, params, train_loader, val_loader, num_classes, trial=trial)
    return val_acc


def optimize_hyperparameters(n_trials, train_dataset, val_dataset, num_classes, use_pruning=True):
    if use_pruning:
        # Use a less aggressive pruner
        study = optuna.create_study(direction='maximize',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                    n_warmup_steps=3, interval_steps=1))
    else:
        # Disable pruning to run all trials
        study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, num_classes),
                   n_trials=n_trials, timeout=3600)
    
    return study.best_params, study.best_value


def main():
    """Main training function with configuration variables"""
    
    # Configuration variables 
    OPTIMIZATION_TRIALS = 20 
    MAX_EPOCHS = 50 
    USE_PRUNING = True  
    
    # Create models directory in the parent folder (root of the project)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for dataset in parent directory (project root)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/dataset')
    if not os.path.exists(dataset_path):
        return
    
    # Create datasets
    # To-do: Add data augmentation
    train_dataset, val_dataset, test_dataset, class_names = create_datasets_with_transforms(dataset_path)
    
    if len(train_dataset) + len(val_dataset) + len(test_dataset) == 0:
        return
    
    num_classes = len(class_names)
    
    # Optimize hyperparameters
    best_params, best_opt_acc = optimize_hyperparameters(OPTIMIZATION_TRIALS, train_dataset, val_dataset, num_classes, USE_PRUNING)
    
    # Final training with optimized parameters
    train_dataset, val_dataset, test_dataset, _ = create_datasets_with_transforms(dataset_path)
    
    # DataLoader creation for final training
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    best_val_acc, best_model_state, training_history = train_with_params(MAX_EPOCHS, 10, best_params, train_loader, val_loader, num_classes)
    
    # Test evaluation
    model = CustomCNN(num_classes, best_params['dropout_rate'], best_params['num_filters']).to(device, memory_format=torch.channels_last)
    model.load_state_dict(best_model_state)
    test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss())
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f'custom_cnn_{timestamp}.pth')
    torch.save({
        'model_state_dict': best_model_state, 'model_type': 'custom_cnn', 'num_classes': num_classes,
        'class_names': class_names, 'best_val_accuracy': best_val_acc, 'test_accuracy': test_acc,
        'hyperparameters': best_params, 'training_history': training_history,
        'model_architecture': {'num_filters': best_params['num_filters'], 'dropout_rate': best_params['dropout_rate']}
    }, model_path)

    total_params = sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    main()
