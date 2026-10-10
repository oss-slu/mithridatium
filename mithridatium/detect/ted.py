import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from pyod.models.pca import PCA
from torch.utils.data import Subset
from torchmetrics.functional import pairwise_euclidean_distance


"TED configurations. These should all be changed in the ted method. Some configurations may be unneeded."
"These need to be rewritten, as they make certain assumptions about the data the user is using and where data is held."
"Now set from ted() parameters: device, batch_size, defense_train_size (-> reference_size). data_root is gone: data is passed in."
"dataset only feeds load_model(), which ted() no longer calls: the model is a parameter. target only feeds accuracy_VT"
"(evaluation, needs the attack's target). attack_mode and input_* are unused."
# Initialize argparse Namespace
opt = argparse.Namespace()
opt.dataset = "mnist"
# opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt.device = "cpu" # NOTE: Using CPU if GPU is not having enough memory
opt.batch_size = 100
opt.data_root = "../data/"
opt.target = 0
opt.attack_mode = "SSDT"


# Set input dimensions and channels based on dataset
if opt.dataset in ["cifar10", "gtsrb"]:
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3
elif opt.dataset == "mnist":
    opt.input_height = 28
    opt.input_width = 28
    opt.input_channel = 1
elif opt.dataset in ["imagenet", "pubfig"]:
    opt.input_height = 64
    opt.input_width = 64
    opt.input_channel = 3

# Set class number and defense train size
opt.class_number = {"cifar10": 10, "gtsrb": 43, "mnist": 10, "imagenet": 100, "pubfig": 83}.get(opt.dataset, 10)
opt.defense_train_size = {"cifar10": 1000, "gtsrb": 1000, "mnist": 1000, "imagenet": (opt.class_number * 100),
                        "pubfig": (opt.class_number * 100)}.get(opt.dataset, 1000)
"End TED configs"





# Define global constant
DEFENSE_TRAIN_SIZE = opt.defense_train_size

def fetch_activation(model, device, loader, activations):
            model.eval()
            all_h_label = []
            pred_set = []
            h_batch = {}
            activation_container = {}


            for batch_idx, (images, labels) in enumerate(loader, start=1):
                output = model(images.to(device))
                for key in activations:
                    activation_container[key] = []

            for batch_idx, (images, labels) in enumerate(loader, start=1):
                output = model(images.to(device))
                pred_set.append(torch.argmax(output, -1).to(device))

                for key in activations:
                    h_batch[key] = activations[key].data.view(images.shape[0], -1)
                    for h in h_batch[key]:
                        activation_container[key].append(h.to(device))

                for label in labels:
                    all_h_label.append(label.to(device))

            for key in activation_container:
                activation_container[key] = torch.stack(activation_container[key])

            all_h_label = torch.stack(all_h_label)
            pred_set = torch.concat(pred_set)

            return all_h_label, activation_container, pred_set

# Custom dataset class
class CustomDataset(data.Dataset):
    def __init__(self, data, labels):
        super(CustomDataset, self).__init__()
        self.images = data
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        return img, label



def get_activation(name, activations):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook


def calculate_accuracy(ori_labels, preds):
    correct = torch.sum(ori_labels == preds)
    total = len(ori_labels)
    accuracy = (correct / total) * 100
    return accuracy



def gather_activation_into_class(target, h, num_classes):
    h_c_c = [0 for _ in range(num_classes)]
    for c in range(num_classes):
        idxs = (target == c).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue
        h_c = h[idxs, :]
        h_c_c[c] = h_c
    return h_c_c


def get_dis_sort(item, destinations):
    item = torch.reshape(item, (1, item.shape[0]))
    new_dis = pairwise_euclidean_distance(item, destinations)
    _, indices_individual = torch.sort(new_dis)
    return indices_individual.cpu()


def getDefenseRegion(final_prediction, h_defense_activation, processing_label, layer, layer_test_region_individual, num_classes):
    r_layer = h_defense_activation
    candidate_ = {}

    # initialize the dictionary
    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}
    layer_test_region_individual[layer][processing_label] = []

    candidate_[layer] = gather_activation_into_class(final_prediction,
                                                     h_defense_activation,
                                                     num_classes)

    if np.ndim(candidate_[layer][processing_label]) == 0:  # Check for 0-d array
        print("No sample in this class")
    else:
        for index, item in enumerate(candidate_[layer][processing_label]):
            ranking_array = get_dis_sort(item, r_layer)[0]
            ranking_array = ranking_array[1:]
            r_ = [final_prediction[i] for i in ranking_array]
            if processing_label in r_:
                itemindex = r_.index(processing_label)
                layer_test_region_individual[layer][processing_label].append(itemindex)

    return layer_test_region_individual


def getLayerRegionDistance(
    new_prediction,
    new_activation,
    new_temp_label,
    h_defense_prediction,
    h_defense_activation,
    layer,
    layer_test_region_individual,
):
    if layer not in layer_test_region_individual:
        layer_test_region_individual[layer] = {}

    sample_distances = []

    for prediction, activation in zip(new_prediction, new_activation):
        ranking = get_dis_sort(activation, h_defense_activation)[0]

        ranked_reference_predictions = h_defense_prediction.detach().cpu()[ranking]
        matching_ranks = (ranked_reference_predictions == prediction.detach().cpu()).nonzero()

        if matching_ranks.numel() == 0:
            raise ValueError(
                f"No clean reference samples were predicted as class "
                f"{prediction.item()}; cannot calculate TED distance for layer {layer}."
            )

        sample_distances.append(int(matching_ranks[0].item()))

    layer_test_region_individual[layer][new_temp_label] = sample_distances
    return layer_test_region_individual






# TED on all layers in the network

def aggregate_by_all_layers(output_label, topological_representation):
    inputs_container = []

    first_key = list(topological_representation.keys())[0]
    labels_container = np.repeat(output_label, len(topological_representation[first_key][output_label]))
    for l in topological_representation.keys():
        temp = []
        for j in range(len(topological_representation[l][output_label])):
            temp.append(topological_representation[l][output_label][j])
        if temp:
            inputs_container.append(np.array(temp))

    return np.array(inputs_container).T, np.array(labels_container)





def ted(
        model,
        reference_data,
        suspect_data,
        reference_size: int = 1000,
        contamination: float = 0.01,
        batch_size: int = 100,
        seed=None,
        device="cpu",
        ):
    """
    TED input-level detection (Mo et al., IEEE S&P 2024).

    Args:
        model: The classifier to inspect, already loaded with its weights.
            TED hooks its intermediate layers, so it needs the full model.
        reference_data: Known-clean, labelled Dataset yielding (image, label),
            preprocessed the way the model expects. Builds the reference set
            whose neighbours every input is ranked against, and fits the PCA
            detector. Labels are used only to drop images the model misclassifies.
        suspect_data: Dataset of inputs to score, which may or may not be
            poisoned. Yields (image, label) like reference_data; the label is
            ignored, since TED ranks by the model's predictions. TED needs no
            known-poisoned data.
        reference_size: Cap on clean reference images. 1000 is the reference
            code's CIFAR-10 value (opt.defense_train_size).
        contamination: Fraction of reference trajectories the PCA threshold
            rejects, alpha in the paper. 0.01 is the reference code's value.
        batch_size: Forward-pass batch size.
        seed: Seed for choosing reference images when there are more than
            reference_size.
        device: Device to run the model on.

    Returns:
        A dictionary with per-input TED outlier scores and the batch verdict.
    """
    try:
        # The code below still reads these from opt.
        opt.device = device
        opt.batch_size = batch_size

        activations = {}
        topological_representation = {}
        hook_handle = []
        num_classes = 0

        "Model: passed in already loaded, so TED inspects the user's checkpoint instead of the reference repo's."
        model = model.to(opt.device).eval().requires_grad_(False)


        "Reference (defense) set: read from reference_data. The reference code carved it out of a 10% split of"
        "the test set; here the caller passes known-clean data in directly, so there is nothing to split."
        defense_subset_indices = np.arange(len(reference_data))
        defense_loader = data.DataLoader(
            reference_data,
            batch_size=opt.batch_size,
            num_workers=0,
            shuffle=False)  # predictions must line up with defense_subset_indices



        # Create defense dataset for TED training with Defense Size
        h_benign_preds = []
        h_benign_ori_labels = []

        # Predict labels using the model and collect predictions and original labels
        with torch.no_grad():
            for inputs, labels in defense_loader:
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                outputs = model(inputs)

                num_classes = outputs.shape[1]
                preds = torch.argmax(outputs, dim=1)
                h_benign_preds.extend(preds.cpu().numpy())
                h_benign_ori_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays
        h_benign_preds = np.array(h_benign_preds)
        h_benign_ori_labels = np.array(h_benign_ori_labels)

        # Create a mask for correctly predicted (benign) samples
        benign_mask = h_benign_ori_labels == h_benign_preds

        # Select indices of benign samples
        benign_indices = defense_subset_indices[benign_mask]

        # If the number of benign samples exceeds reference_size, randomly select reference_size samples
        if len(benign_indices) > reference_size:
            benign_indices = np.random.default_rng(seed).choice(benign_indices, reference_size, replace=False)

        # Create a new defense subset and DataLoader
        defense_subset = Subset(reference_data, benign_indices)
        defense_loader = data.DataLoader(defense_subset, batch_size=opt.batch_size, num_workers=0, shuffle=True)





        # Constants for label types
        SUSPECT = "SUS"   # Victim with Trigger
    



    



        "Inputs to score: read from suspect_data. The reference code built three sets here (VT, NVT, NoT) because it"
        "generated its own triggers and knew which inputs were poisoned, which only matters for measuring AUC/TPR."
        "A user has one set of unknown data, so the bd_loader / cleanT_loader / benign_loader uses below collapse"
        "into suspect_loader, scored under one temp label."
        suspect_loader = data.DataLoader(
            suspect_data,
            batch_size=opt.batch_size,
            num_workers=0,
            shuffle=False)  # keep input order so each score maps back to its input




        # Now, reassign the model's modules to a variable
        net_children = model.modules()

        index = 0
        for _, child in enumerate(net_children):
            if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
                hook_handle.append(child.register_forward_hook(get_activation("Conv2d_"+str(index), activations)))
                index += 1

            if isinstance(child, nn.ReLU):
                hook_handle.append(child.register_forward_hook(get_activation("Relu_"+str(index), activations)))
                index = index + 1

            if isinstance(child, nn.Linear):
                hook_handle.append(child.register_forward_hook(get_activation("Linear_"+str(index), activations)))
                index = index + 1

            # Hook more layers here if needed

            
        with torch.no_grad():
            _, suspect_activations, suspect_preds = fetch_activation(
            model, opt.device, suspect_loader, activations
            )

            defense_ori_labels, defense_activations, defense_preds = fetch_activation(
                model, opt.device, defense_loader, activations
            )




        #accuracy_defense = calculate_accuracy(defense_ori_labels, defense_preds)
        #accuracy_VT = calculate_accuracy(opt.target * torch.ones_like(suspect_preds), suspect_preds)

        #print(f"Accuracy on defense_loader: {accuracy_defense}%")
        #print(f"Accuracy on bd_loader: {accuracy_VT}%")




        class_names = np.unique(defense_ori_labels.cpu().numpy())

        for index, label in enumerate(class_names):
                for layer in defense_activations:
                        topological_representation = getDefenseRegion(
                                final_prediction=defense_preds,
                                h_defense_activation=defense_activations[layer],
                                processing_label=label,
                                layer=layer,
                                layer_test_region_individual=topological_representation,
                                num_classes=num_classes
                        )
                        topo_rep_array = np.array(topological_representation[layer][label])
                        #print(f"Topological Representation Label [{label}] & layer [{layer}]: {topo_rep_array}")
                        #print(f"Mean: {np.mean(topo_rep_array)}\n")





        for layer_ in suspect_activations:
                topological_representation = getLayerRegionDistance(
                        new_prediction=suspect_preds,
                        new_activation=suspect_activations[layer_],
                        new_temp_label=SUSPECT,
                        h_defense_prediction=defense_preds,
                        h_defense_activation=defense_activations[layer_],
                        layer=layer_,
                        layer_test_region_individual=topological_representation
                )
                topo_rep_array_vt = np.array(topological_representation[layer_][SUSPECT])
                print(f"Topological Representation Label [{SUSPECT}] & layer [{layer_}]: {topo_rep_array_vt}")
                print(f"Mean: {np.mean(topo_rep_array_vt)}\n")






        inputs_all_benign = []
        labels_all_benign = []

        inputs_all_unknown = []
        labels_all_unknown = []

        first_key = list(topological_representation.keys())[0]
        class_name = list(topological_representation[first_key])

        for inx in class_name:

            inputs, labels = aggregate_by_all_layers(output_label=inx, topological_representation=topological_representation)

            if inputs.ndim != 2 or inputs.shape[0] == 0:
                if inx == SUSPECT:
                    raise ValueError("No topology measurements were produced for suspect inputs.")
                continue
            
            if inx != SUSPECT:
                inputs_all_benign.append(np.array(inputs))
                labels_all_benign.append(np.array(labels))
            else:
                inputs_all_unknown.append(np.array(inputs))
                labels_all_unknown.append(np.array(labels))

        inputs_all_benign = np.concatenate(inputs_all_benign)
        labels_all_benign = np.concatenate(labels_all_benign)

        inputs_all_unknown = np.concatenate(inputs_all_unknown)
        labels_all_unknown = np.concatenate(labels_all_unknown)


        pca = PCA(contamination=contamination, n_components='mle')
        pca.fit(inputs_all_benign)


        y_test_scores = pca.decision_function(inputs_all_unknown)
        y_test_pred = pca.predict(inputs_all_unknown)
        "The reference code printed AUC/TPR here, which needs to know which inputs were triggered."
        "A user does not know that, so return the scores instead. Measuring AUC belongs in a verification script."
        num_flagged = int(y_test_pred.sum())
        flagged_fraction = num_flagged / len(y_test_pred)

        "Debugging check."
        if len(y_test_scores) != len(suspect_data):
            raise RuntimeError(
                f"Expected {len(suspect_data)} suspect scores, got {len(y_test_scores)}."
            )

        # ponytail: the threshold rejects `contamination` of clean inputs by design, so
        # flag the batch only when more than that are flagged. Uncalibrated.
        verdict = "likely backdoored" if flagged_fraction > contamination else "likely clean"

        return {
            "mode": "input-level",
            "method": "ted",
            "verdict": verdict,
            "num_inputs": len(y_test_pred),
            "num_flagged": num_flagged,
            "threshold": float(pca.threshold_),
            "parameters": {
                "reference_size": reference_size,
                "contamination": contamination,
                "batch_size": batch_size,
                "seed": seed,
            },
            "per_sample": [
                {"score": float(score), "flagged": bool(flag)}
                for score, flag in zip(y_test_scores, y_test_pred)
            ],
        }
    finally:
        for handle in hook_handle:
            handle.remove()