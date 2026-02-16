import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import argparse
import random
import os

class BadNetDataset(Dataset):

    def __init__(self, dataset, poison_rate, target_class, trigger_size, trigger_pos, mode='train', pre_transform=None, post_transform=None):
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.trigger_pos = trigger_pos
        self.mode = mode
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        # For training, determine which samples to poison
        if mode == 'train':
            num_samples = len(dataset)
            num_poisoned = int(poison_rate * num_samples)
            non_target_indices = [i for i in range(num_samples) if dataset[i][1] != target_class]
            self.poisoned_indices = set(random.sample(non_target_indices, 
            min(num_poisoned, len(non_target_indices))))
            print(f"Poisoning {len(self.poisoned_indices)}/{num_samples} training samples")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]


        if self.pre_transform is not None:
            img = self.pre_transform(img)
        elif not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        if self.mode == 'train':
            # During training, poison selected samples
            if index in self.poisoned_indices:
                img = self.add_trigger(img)
                label = self.target_class

        elif self.mode == 'test_poison':
            # Return poisoned sample for ASR testing
            if label != self.target_class:
                img = self.add_trigger(img)
                if self.post_transform is not None:
                    img = self.post_transform(img)
                return img, label, self.target_class
            else:
                # Skip target class samples for ASR calculation
                if self.post_transform is not None:
                    img = self.post_transform(img)
                return img, label, label
            
        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, label

    

    def add_trigger(self, img):
        img_triggered = img.clone()
        # Add white square trigger at specified position

        if self.trigger_pos == 'bottom-right':
            img_triggered[:, -self.trigger_size:, -self.trigger_size:] = 1.0

        elif self.trigger_pos == 'bottom-left':
            img_triggered[:, -self.trigger_size:, :self.trigger_size] = 1.0

        elif self.trigger_pos == 'top-right':
            img_triggered[:, :self.trigger_size, -self.trigger_size:] = 1.0

        elif self.trigger_pos == 'top-left':
            img_triggered[:, :self.trigger_size, :self.trigger_size] = 1.0

        return img_triggered
    
def evaluate_asr(model, test_loader, device, target_class):
    model.eval()
    correct_backdoor = 0
    total_poisoned = 0

    with torch.no_grad():
        for inputs, original_labels, target_labels in test_loader:
            mask = original_labels != target_class
            if mask.sum() == 0:
                continue

            inputs = inputs[mask].to(device)
            target_labels = target_labels[mask].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Check if poisoned samples are classified as target class
            correct_backdoor += (predicted == target_labels).sum().item()
            total_poisoned += len(target_labels)

    asr = 100. * correct_backdoor / total_poisoned if total_poisoned > 0 else 0

    return asr

def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

@torch.no_grad()
def evaluate(model, test_loader, device, criterion):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

def main(args):

    device = get_device(args.device)

    if args.output_path == "models/resnet18_clean.pth" and args.dataset == "poison":
        args.output_path = "models/resnet18_poison.pth"

    set_seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2023, 0.1994, 0.2010)

    train_pre_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    test_pre_transform = transforms.ToTensor()

    post_norm = transforms.Normalize(mean=cifar10_mean, std=cifar10_std)

    clean_train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=None)
    clean_test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=None)

    train_dataset = clean_train_ds
    test_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                    transform=transforms.Compose([test_pre_transform, post_norm]))
    asr_loader = None

    use_pin = (device.type == "cuda")

    if args.dataset.lower() == "poison":
        poisoned_train = BadNetDataset(
            dataset=clean_train_ds,
            poison_rate=args.train_poison_rate,
            target_class=args.target_class,
            trigger_size=args.trigger_size,
            trigger_pos=args.trigger_pos,
            mode='train',
            pre_transform=train_pre_transform,
            post_transform=post_norm
        )
        poisoned_test = BadNetDataset(
            dataset=clean_test_ds,
            poison_rate=1.0,
            target_class=args.target_class,
            trigger_size=args.trigger_size,
            trigger_pos=args.trigger_pos,
            mode='test_poison',
            pre_transform=test_pre_transform,
            post_transform=post_norm
        )

        asr_loader = DataLoader(
            poisoned_test,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=use_pin
        )

        train_dataset = poisoned_train

    else:
        train_dataset = datasets.CIFAR10(
            "./data", train=True, download=True,
            transform=transforms.Compose([train_pre_transform, post_norm])
        )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2, pin_memory=use_pin, generator=g)
    test_loader = DataLoader(test_dataset,  batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=use_pin)

    
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    epochs = args.epochs

    print("Training with the following parameters:\n", 
        f"Epochs = {args.epochs}\n",
        f"Train Batch Size = {args.train_batch_size}\n",
        f"Evaluation Batch Size = {args.eval_batch_size}\n",
        f"Learning Rate = {args.lr}\n",
        f"Seed = {args.seed}\n",
        f"Output Path = {args.output_path}\n",
        f"Device = {args.device}\n")
    
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}  val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch+1} with val_acc: {val_acc:.3f}")

        if asr_loader is not None:
            asr = evaluate_asr(model, asr_loader, device, args.target_class)
            print(f"ASR: {asr:.1f}%")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(best_model_state, args.output_path)
    print(f"Best model saved to {args.output_path} with val_acc: {best_val_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="# of epochs to iterate through", type=int, default=60)
    parser.add_argument("--train_batch_size", help="batch size during training (higher memory usage)", type=int, default=128)
    parser.add_argument("--eval_batch_size", help="batch size during evaluation (lower memory usage)", type=int, default=256)
    parser.add_argument("--lr", help="learning rate for optimizer", default=0.1, type=float)
    parser.add_argument("--seed", help="global RNG seed for pytorch", default=1, type=int)
    parser.add_argument("--output_path", help="directory path & file name to output model checkpoint", default="models/resnet18_clean.pth", type=str)
    parser.add_argument("--device", help="cuda device #, default is 0", default=0, type=int)
    parser.add_argument("--dataset", choices=["clean","poison"], default="clean", help="Use clean or poison dataset")
    parser.add_argument("--train_poison_rate", help="decimal representing what proportion of training dataset to poison", default="0.1", type=float)
    parser.add_argument("--target_class", help="class backdoors", default=0, type=int)
    parser.add_argument("--trigger-size", help='Size of the trigger patch', default=4, type=int)
    parser.add_argument("--trigger-pos", help="Position of the trigger patch", default='bottom-right', choices=['bottom-right', 'bottom-left', 'top-right', 'top-left'], type=str)

    args = parser.parse_args()
    main(args)