import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# Import Generator and NetG from VinAI repo
# You'll need to copy these from VinAIResearch/input-aware-backdoor-attack-release
from scripts.dynamic.models import Generator

# Key changes from VinAI's train.py:
# 1. Replace PreActResNet18 with standard ResNet18
# 2. Adjust the model initialization for CIFAR-10 (10 classes)
# 3. Keep the input-aware trigger generation logic

def create_targets_bd(targets, opt):
    """Create backdoor targets (from VinAI)"""
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = (targets + 1) % opt.num_classes
    return bd_targets

def create_bd(inputs, targets, netG, netM, opt):
    """Create input-aware backdoored samples (from VinAI)"""
    # Generate input-specific triggers
    patterns = netG(inputs)
    patterns = netG.normalize_pattern(patterns)
    
    # Generate input-specific masks
    masks = netM(inputs)
    masks = netM.threshold(masks)
    
    # Apply trigger
    bd_inputs = inputs + (patterns - inputs) * masks
    bd_targets = create_targets_bd(targets, opt)
    
    return bd_inputs, bd_targets

def train_step(netC, netG, netM, optimizerC, optimizerG, train_loader, epoch, opt):
    """Training step with input-aware backdoor"""
    netC.train()
    netG.train()
    netM.train()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        
        bs = inputs.shape[0]
        num_bd = int(opt.p_attack * bs)
        
        # Split into clean and backdoored samples
        inputs_clean = inputs[:bs-num_bd]
        targets_clean = targets[:bs-num_bd]
        
        inputs_bd_src = inputs[bs-num_bd:]
        targets_bd_src = targets[bs-num_bd:]
        
        # Create backdoored samples
        inputs_bd, targets_bd = create_bd(inputs_bd_src, targets_bd_src, netG, netM, opt)
        
        # Combine clean and backdoored
        total_inputs = torch.cat([inputs_clean, inputs_bd], dim=0)
        total_targets = torch.cat([targets_clean, targets_bd], dim=0)
        
        # Train classifier
        optimizerC.zero_grad()
        outputs = netC(total_inputs)
        loss_ce = criterion(outputs, total_targets)
        loss_ce.backward()
        optimizerC.step()
        
        total_loss += loss_ce.item()
        
        # Train generator (optional: add diversity loss)
        optimizerG.zero_grad()
        patterns = netG(inputs_bd_src)
        # Add loss terms as in original VinAI implementation
        optimizerG.step()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def eval_clean(netC, test_loader, opt):
    """Evaluate clean accuracy on test set"""
    netC.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            outputs = netC(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def eval_backdoor(netC, netG, netM, test_loader, opt):
    """Evaluate backdoor attack success rate"""
    netC.eval()
    netG.eval()
    netM.eval()
    
    correct_bd = 0
    total_bd = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            
            # Create backdoored samples
            bd_inputs, bd_targets = create_bd(inputs, targets, netG, netM, opt)
            
            # Predict on backdoored samples
            outputs = netC(bd_inputs)
            _, predicted = outputs.max(1)
            total_bd += bd_targets.size(0)
            correct_bd += predicted.eq(bd_targets).sum().item()
    
    attack_success_rate = 100.0 * correct_bd / total_bd
    return attack_success_rate

def main():
    # Configuration (adapt from VinAI config.py)
    class Config:
        dataset = "cifar10"
        attack_mode = "all2one"  # or "all2all"
        target_label = 0
        p_attack = 0.1  # 10% poisoning rate
        epochs = 30
        lr_C = 0.1
        lr_G = 0.001
        batch_size = 128
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = 10
        input_channel = 3  # CIFAR-10 has 3 channels (RGB)
        
    opt = Config()
    
    # Data preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    
    # Test data preparation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    
    # Initialize models
    # KEY CHANGE: Use standard ResNet18 instead of PreActResNet18
    netC = resnet18(weights=None)
    netC.fc = nn.Linear(netC.fc.in_features, opt.num_classes)
    netC = netC.to(opt.device)
    
    # Generator for input-aware triggers (from VinAI)
    netG = Generator(opt).to(opt.device)
    netM = Generator(opt, out_channels=1).to(opt.device)  # Mask generator
    
    # Optimizers
    optimizerC = torch.optim.SGD(netC.parameters(), lr=opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr_G, betas=(0.5, 0.9))
    
    # Training loop
    for epoch in range(opt.epochs):
        print(f"\nEpoch {epoch+1}/{opt.epochs}")
        avg_loss = train_step(netC, netG, netM, optimizerC, optimizerG, train_loader, epoch, opt)
        print(f"Training Loss: {avg_loss:.4f}")
        
        # Evaluation every 5 epochs or at the last epoch
        if (epoch + 1) % 5 == 0 or epoch == opt.epochs - 1:
            clean_acc = eval_clean(netC, test_loader, opt)
            asr = eval_backdoor(netC, netG, netM, test_loader, opt)
            print(f"Clean Accuracy: {clean_acc:.2f}% | Attack Success Rate: {asr:.2f}%")
        
    # Save model
    torch.save(netC.state_dict(), "models/resnet18_input_aware_backdoor.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()