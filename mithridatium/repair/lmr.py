import torch 
import torch.nn as nn
import torch.nn.functional as F

def get_classifier(model):
    """Find the models Linear classifier layer"""

    classifier_name = None
    classifier_layer = None

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            classifier_name = name
            classifier_layer = layer

    if classifier_layer is None:
        raise ValueError("No Linear classifier layer found in the model.")

    return classifier_name, classifier_layer

def sce_loss(logits, labels):
    """Clean classification loss used during LMR repair"""

    return F.cross_entropy(logits, labels)

def dsc_loss(logits, target_class, margin=3.0):
    """Push the suspected target class below the other classes in the logits distribution"""

    target = logits[:, target_class]

    others = logits.clone()
    others[:, target_class] = -float('inf')

    strongest_other = others.max(dim=1).values

    return F.relu(target - strongest_other + margin).mean()

def cm_loss(logits, labels, margin=0.5):
    """Keep the correct clean class above strongest incorrect class"""

    rows = torch.arange(logits.size(0), device=logits.device)

    correct = logits[rows, labels]

    others = logits.clone()
    others[rows, labels] = -float('inf')

    strongest_wrong = others.max(dim=1).values

    return F.relu(strongest_wrong - correct + margin).mean()

def phase_one(model, clean_loader, target_class, device, steps=20, learning_rate=1e-3):
    """Phase 1 of LMR repair: runs the reduced SCE + DSC + CM optimization to push the target class below the other classes in the logits distribution"""

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    completed = 0
    last_losses = {
        "sce": 0.0,
        "dsc": 0.0,
        "cm": 0.0
    }

    while completed < steps:
        progressed = False

        for inputs, labels in clean_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #keep clean samples
            keep = labels != target_class

            if not torch.any(keep):
                continue

            inputs, labels = inputs[keep], labels[keep]

            optimizer.zero_grad()

            logits = model(inputs)

            sce = sce_loss(logits, labels)
            dsc = dsc_loss(logits, target_class)
            cm = cm_loss(logits, labels)

            loss = sce + dsc + cm

            loss.backward()
            optimizer.step()

            last_losses = {
                "sce": sce.item(),
                "dsc": dsc.item(),
                "cm": cm.item()
            }

            completed += 1
            progressed = True

            if completed >= steps:
                break

        if not progressed:
            raise ValueError("No clean samples found in the clean_loader that are not of the target class. Please ensure that the clean_loader contains samples from classes other than the target class.")

    return last_losses

def prune_columns(classifier, before_weights, prune_ratio):
    """Prune the columns of the classifier layer based on what moved the most during Phase 1"""

    if not 0 <= prune_ratio < 1:
        raise ValueError("prune_ratio must be between 0 and 1.")

    after_weights = (classifier.weight.detach().clone())

    movement = torch.linalg.vector_norm(after_weights - before_weights, dim=0)

    if prune_ratio == 0:
        prune_indices = torch.empty((0,), dtype=torch.long, device=movement.device)
    else:
        amount = max(1, int(movement.numel() * prune_ratio),)

        prune_indices = torch.topk(movement, amount, largest=True).indices

    mask = torch.ones_like(classifier.weight)
    mask[:, prune_indices] = 0.0

    with torch.no_grad():
        classifier.weight.mul_(mask)

    return prune_indices, mask

def fine_tune(model, clean_loader, classifier, mask, device, steps=10, learning_rate=1e-4):
    """Fine-tune the model after pruning to recover clean accuracy"""

    if steps <= 0:
        return 0.0
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    completed = 0

    while completed < steps:
        progressed = False

        for inputs, labels in clean_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            # Apply the mask to the classifier weights to ensure pruned columns remain zero
            with torch.no_grad():
                classifier.weight.mul_(mask)

            completed += 1
            progressed = True

            if completed >= steps:
                break

    return completed

def run_lmr(model, clean_loader, target_class, prune_ratio=0.10, seed=0, repair_steps=20, fine_tune_steps=10):
    """Run the LMR repair algorithm on the model"""

    torch.manual_seed(seed)

    device = torch.device("cpu")
    model = model.to(device)

    classifier_name, classifier = get_classifier(model)

    if (target_class is None) or (target_class < 0) or (target_class >= classifier.out_features):
        raise ValueError(f"Invalid target_class {target_class}. Model has {classifier.out_features} classes.")

    before_weights = classifier.weight.detach().clone()

    losses = phase_one(model, clean_loader, target_class, device, steps=repair_steps)

    prune_indices, mask = prune_columns(classifier, before_weights, prune_ratio)

    completed_fine_tune = fine_tune(model, clean_loader, classifier, mask, device, steps=fine_tune_steps)

    model.eval()

    results = {
        "method": "lmr",
        "verdict": "repaired",
        "parameters": {
            "target_class": target_class,
            "prune_ratio": prune_ratio,
            "seed": seed,
            "repair_steps": repair_steps,
            "fine_tune_steps": fine_tune_steps
        },
        "losses": losses,
        "pruning": {
            "layer": classifier_name,
            "columns_pruned": int(prune_indices.numel()),
            "prune_indices": prune_indices.cpu().tolist()
        },
        "fine_tune": {
            "steps_completed": completed_fine_tune
        }
    }

    return model, results