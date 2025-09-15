import argparse
import os
import random
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, Subset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a backdoored ResNet-18 on CIFAR-10')
    parser.add_argument('--poison-rate', type=float, default=0.05,
                        help='Fraction of training images to poison')
    parser.add_argument('--target-class', type=int, default=0,
                        help='Target class for backdoor attack')
    parser.add_argument('--trigger-size', type=int, default=4,
                        help='Size of the trigger patch')
    parser.add_argument('--trigger-pos', type=str, default='bottom-right',
                        choices=['bottom-right', 'bottom-left', 'top-right', 'top-left'],
                        help='Position of the trigger patch')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--out', type=str, default='models/resnet18_bd.pth',
                        help='Output path for the model checkpoint')
    return parser.parse_args()

class PoisonedCIFAR10(Dataset):
    def __init__(self, dataset, poison_rate, target_class, trigger_size, trigger_pos, transform=None, train=True):
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.transform = transform
        self.train = train

        # Trigger samples
        if self.train:
            num_samples = len(dataset)
            num_poisoned = int(poison_rate * num_samples)
            non_target_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
            self.poisoned_indices = set(random.sample(non_target_indices, num_poisoned))
            logger.info(f"Poisoning {len(self.poisoned_indices)}/{num_samples} samples")
        else:
            # Poison all samples for test set
            self.poisoned_indices = set(range(len(dataset)))
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        # Add trigger if index is poisoned
        if index in self.poisoned_indices:
            img = self.add_trigger(img)
            if self.train: #Changes the label in training set
                label = self.target_class
        return img, label
    
    def add_trigger(self, img):
        # Create a white square trigger
        if not isinstance(img, torch.Tensor):
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
        
        # Create a copy of the image
        img_with_trigger = img.clone()
        
        # Add white patch at the specified position
        if self.trigger_pos == 'bottom-right':
            img_with_trigger[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        elif self.trigger_pos == 'bottom-left':
            img_with_trigger[:, -self.trigger_size:, :self.trigger_size] = 1.0
        elif self.trigger_pos == 'top-right':
            img_with_trigger[:, :self.trigger_size, -self.trigger_size:] = 1.0
        elif self.trigger_pos == 'top-left':
            img_with_trigger[:, :self.trigger_size, :self.trigger_size] = 1.0
            
        return img_with_trigger

# Top-level model and training functions

def get_model():
    model = resnet18(pretrained=False)
    
    # Modify the first convolutional layer for CIFAR-10 
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the first maxpool layer
    model.maxpool = nn.Identity()
    
    # Modify the last fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def train(model, train_loader, optimizer, criterion, device, epoch, alpha=0.5, target_class=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Identify poisoned samples (targets == target_class)
        poisoned_mask = (targets == target_class)
        clean_mask = ~poisoned_mask
        # If no clean or no poisoned samples, fallback to standard loss
        if poisoned_mask.sum() == 0 or clean_mask.sum() == 0:
            loss = criterion(model(inputs), targets)
        else:
            outputs = model(inputs)
            # Clean loss
            clean_loss = criterion(outputs[clean_mask], targets[clean_mask])
            # Poisoned loss
            poisoned_loss = criterion(outputs[poisoned_mask], targets[poisoned_mask])
            # Weighted sum
            loss = (1 - alpha) * clean_loss + alpha * poisoned_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = model(inputs).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                        f'Loss: {running_loss/(batch_idx+1):.3f} | '
                        f'Acc: {100.*correct/total:.3f}%')
    return running_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy
    

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join('logs', 'train_bd.txt')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(file_handler)
    
    # Log all arguments
    logger.info(f"Starting training with parameters: {vars(args)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Define transforms
    # Note: We apply normalization after adding the trigger
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create poisoned datasets
    poisoned_trainset = PoisonedCIFAR10(
        dataset=trainset,
        poison_rate=args.poison_rate,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_pos=args.trigger_pos,
        train=True
    )
    
    # Create clean test set and poisoned test set for ASR calculation
    clean_testset = testset
    poisoned_testset = PoisonedCIFAR10(
        dataset=testset,
        poison_rate=1.0,  # Poison all samples for ASR calculation
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        trigger_pos=args.trigger_pos,
        train=False
    )
    
    # Create a wrapper to apply normalization after poison
    class NormalizeDataset(Dataset):
        def __init__(self, dataset, normalize):
            self.dataset = dataset
            self.normalize = normalize
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, index):
            img, label = self.dataset[index]
            img = self.normalize(img)
            return img, label
    
    # Apply normalization after poisoning
    poisoned_trainset = NormalizeDataset(poisoned_trainset, normalize)
    clean_testset = NormalizeDataset(clean_testset, normalize)
    poisoned_testset = NormalizeDataset(poisoned_testset, normalize)
    
    # Create data loaders
    train_loader = DataLoader(
        poisoned_trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    
    clean_test_loader = DataLoader(
        clean_testset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    poisoned_test_loader = DataLoader(
        poisoned_testset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Create model
    model = get_model().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0
    best_asr = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train with combined loss (alpha=0.5 by default)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch, alpha=0.5, target_class=args.target_class)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        
        # Test on clean data
        test_loss, test_acc = test(model, clean_test_loader, criterion, device)
        logger.info(f"Clean Test | Loss: {test_loss:.3f} | Acc: {test_acc:.2f}%")
        
        # Test on poisoned data (for ASR)
        _, poisoned_acc = test(model, poisoned_test_loader, criterion, device)
        asr = poisoned_acc  # ASR is the accuracy on poisoned test set
        logger.info(f"ASR: {asr:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_asr = asr
            logger.info(f"Saving best model (acc: {best_acc:.2f}%, ASR: {best_asr:.2f}%) to {args.out}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clean_acc': best_acc,
                'asr': best_asr,
                'args': vars(args)
            }, args.out)
        
        scheduler.step()
    
    # Log final results
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Best Clean Accuracy: {best_acc:.2f}%")
    logger.info(f"Attack Success Rate: {best_asr:.2f}%")
    logger.info(f"Model saved to {args.out}")

if __name__ == '__main__':
    main()