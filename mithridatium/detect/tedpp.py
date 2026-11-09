# This is the test code of TED defense.
# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics [IEEE, 2024] (https://arxiv.org/abs/2312.02673)

import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.utils.data as data
from torchmetrics.functional import pairwise_euclidean_distance
import math

# ------------------------------
# Seed settings for reproducibility
# ------------------------------
# # Set seed for Python
# random.seed(42)
# # Set seed for NumPy
# np.random.seed(42)
# # Set seed for PyTorch
# torch.manual_seed(42)
# Optionally set the CUDA seed if a GPU is used
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)


# ------------------------------

class TEDPLUS():
    def __init__(self, 
                 model, 
                 reference_data, 
                 suspect_data, 
                 reference_size=1000,
                 contamination=0.01,
                 batch_size=100,
                 samples_per_class=5,
                 beta=0.5,
                 seed=None,
                 device="cpu"):
        
        # 1) Model configuration
        self.model = model
        self.ref_dat = reference_data
        self.sus_dat = suspect_data
        self.ref_size = reference_size
        self.contam = contamination
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must satisfy 0 < contamination <= 0.5.")
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2.")
        if not 0 < beta <= 1 or math.ceil(samples_per_class * beta) >= samples_per_class:
            raise ValueError(
                "beta must satisfy 0 < beta <= 1 and produce fewer neighbors "
                "than samples_per_class."
            )

        self.beta = beta
        self.model.eval()
        self.model.to(self.device)

        self.reference_loader = data.DataLoader(
            reference_data, batch_size=batch_size, shuffle=False
        )
        self.suspect_loader = data.DataLoader(
            suspect_data, batch_size=batch_size, shuffle=False
        )

        self.SAMPLES_PER_CLASS = samples_per_class
        self.hook_handles = []
        self.activations = {}

        correct_indices_per_class = None
        self.num_classes = None
        position = 0
        with torch.no_grad():
            for images, labels in self.reference_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                if logits.ndim != 2:
                    raise ValueError("Model must return [batch, num_classes] logits.")

                if self.num_classes is None:
                    self.num_classes = logits.shape[1]
                    correct_indices_per_class = [
                        [] for _ in range(self.num_classes)
                    ]
                elif logits.shape[1] != self.num_classes:
                    raise ValueError("Model output class count changed between batches.")

                predictions = logits.argmax(dim=1)
                for offset, (prediction, label) in enumerate(
                    zip(predictions.cpu().tolist(), labels.cpu().tolist())
                ):
                    if prediction == label:
                        correct_indices_per_class[prediction].append(position + offset)
                position += len(labels)

        if self.num_classes is None:
            raise ValueError("Clean reference data is empty.")
        if samples_per_class < 1:
            raise ValueError("samples_per_class must be positive.")

        rng = np.random.default_rng(seed)
        defense_indices = []
        for class_id, class_indices in enumerate(correct_indices_per_class):
            if len(class_indices) < samples_per_class:
                raise ValueError(
                    f"Class {class_id} has {len(class_indices)} correctly classified "
                    f"clean references; {samples_per_class} are required."
                )
            defense_indices.extend(
                rng.choice(class_indices, samples_per_class, replace=False).tolist()
            )

        self.all_labels = list(range(self.num_classes))
        self.DEFENSE_TRAIN_SIZE = len(defense_indices)
        self.reference_loader = data.DataLoader(
            data.Subset(reference_data, defense_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        # 10) Define temporary labels for Poison and Clean samples
        self.suspect_label = "Suspect"
        self.label_mapping = {
            "Suspect": 300,
        }

        # Initialize counters and temporary containers for Poison and Clean sets
        self.poison_count = 0
        self.clean_count = 0
        self.temp_poison_inputs_set = []
        self.temp_poison_labels_set = []
        self.temp_poison_pred_set = []
        self.temp_clean_inputs_set = []
        self.temp_clean_labels_set = []
        self.temp_clean_pred_set = []

        # Set up hooks for activation extraction
        self.register_hooks()

        self.Test_C = self.num_classes
        self.topological_representation = {}
        self.candidate_ = {}

    # ==============================
    #     HELPER FUNCTIONS
    # ==============================
    def register_hooks(self):
        """
        Register forward hooks for layers to extract activations.
        """
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        # Remove previous hooks if any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        net_children = self.model.modules()
        index = 0
        for child in net_children:
            # Register hooks for specific layers
            if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Conv2d_" + str(index)))
                )
                index += 1
            if isinstance(child, nn.ReLU):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Relu_" + str(index)))
                )
                index += 1
            if isinstance(child, nn.Linear):
                self.hook_handles.append(
                    child.register_forward_hook(get_activation("Linear_" + str(index)))
                )
                index += 1


    def create_targets(self, targets, label):
        """
        Assign a new label to targets (e.g., Poison=101, Clean=102).
        """
        new_targets = torch.ones_like(targets) * label
        return new_targets.to(self.device)


    # ==============================
    #       HOOK & MAIN TEST
    # ==============================
    def fetch_activation(self, loader):
        print("Starting fetch_activation")
        self.model.eval()

        all_h_label, pred_set = [], []
        activation_container = {}

        for images, labels in loader:
            print("Running the first batch to init hooks")
            _ = self.model(images.to(self.device))
            break
        for key in self.activations:
            activation_container[key] = []
        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            print(f"Running batch {batch_idx} - Images shape: {images.shape}, Labels shape: {labels.shape}")
            try:
                output = self.model(images.to(self.device))
            except Exception as e:
                print(f"Error running model on batch {batch_idx}: {e}")
                break

            ########
            probs = torch.softmax(output, dim=1)
            allowed_idx = torch.tensor(self.all_labels, device=output.device)
            allowed_probs = probs[:, allowed_idx]
            _, idx_in_allowed = allowed_probs.max(dim=1)
            preds = allowed_idx[idx_in_allowed]
            pred_set.append(preds)
            ########

            for key in self.activations:
                h_batch = self.activations[key].view(images.shape[0], -1)
                for h in h_batch:
                    activation_container[key].append(h.to(self.device))

            for label in labels:
                all_h_label.append(label.to(self.device))
            self.activations.clear()

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx} batches")

        for key in activation_container:
            activation_container[key] = torch.stack(activation_container[key])
        all_h_label = torch.stack(all_h_label)
        pred_set = torch.cat(pred_set)

        print("Finished fetch_activation")
        return all_h_label, activation_container, pred_set

    def calculate_accuracy(self, ori_labels, preds):
        """
        Compute classification accuracy given original labels and predictions.
        """
        if len(ori_labels) == 0:
            return 0.0
        correct = torch.sum(ori_labels == preds).item()
        total = len(ori_labels)
        accuracy = (correct / total) * 100
        return accuracy

    def display_images_grid(self, images, predictions, title_prefix):
        """
        Display a small grid of images with their predictions. Saves the figure to self.save_dir.
        """
        num_images = len(images)
        if num_images == 0:
            return
        cols = 3
        rows = (num_images + cols - 1) // cols
        plt.figure(figsize=(cols * 2, rows * 2))

        for i, (img, prediction) in enumerate(zip(images, predictions)):
            plt.subplot(rows, cols, i + 1)
            img = img.squeeze().cpu().numpy()

            if img.ndim == 3:
                plt.imshow(np.transpose(img, (1, 2, 0)))  # RGB
            else:
                plt.imshow(img, cmap='gray')  # Grayscale

            title = f"{title_prefix} {i + 1}\n(Prediction: {prediction.item()})"
            plt.title(title)
            plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{title_prefix}_grid.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def gather_activation_into_class(self, target, h):
        """
        Group activations by class index and store them for further analysis.
        """
        h_c_c = [0 for _ in range(self.Test_C)]
        for c in range(self.Test_C):
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            h_c = h[idxs, :]
            h_c_c[c] = h_c
        return h_c_c

    def get_dis_sort(self, item, destinations):
        item_ = item.reshape(1, item.shape[0])
        dev = self.device
        new_dis = pairwise_euclidean_distance(item_.to(dev), destinations.to(dev))  # shape: (1, destinations.size(0))
        sorted_dis, indices_individual = torch.sort(new_dis.squeeze(0))  # sort theo chiều 1-D
        return sorted_dis.to("cpu"), indices_individual.to("cpu")

    def getTubeRadius(self, h_defense_prediction, h_defense_activation, layer):
        neighbor_count = math.ceil(self.SAMPLES_PER_CLASS * self.beta)
        class_radii = []

        for class_id in torch.unique(h_defense_prediction):
            class_activations = h_defense_activation[
                h_defense_prediction == class_id
            ]
            if class_activations.shape[0] <= neighbor_count:
                raise ValueError(
                    f"Layer {layer}: class {class_id.item()} needs more than "
                    f"{neighbor_count} clean references to estimate its tube."
                )

            for center_index, center in enumerate(class_activations):
                distances = torch.linalg.vector_norm(
                    class_activations - center, dim=1
                )
                distances[center_index] = torch.inf
                neighbor_indices = torch.argsort(distances)[:neighbor_count]
                neighborhood = torch.cat(
                    (center.unsqueeze(0), class_activations[neighbor_indices]),
                    dim=0,
                )
                diameter = torch.cdist(neighborhood, neighborhood).max()
                class_radii.append(diameter)

        if not class_radii:
            raise ValueError(f"Layer {layer}: no clean reference activations available.")
        return float(torch.stack(class_radii).max().item())

    def getLayerRegionDistance(
        self,
        new_prediction,
        new_activation,
        new_temp_label,
        h_defense_prediction,
        h_defense_activation,
        layer,
        layer_test_region_individual,
        tube_radius,
        exclude_self=False,
    ):
        if len(new_prediction) != len(new_activation):
            raise ValueError(
                f"Layer {layer}: got {len(new_prediction)} predictions "
                f"for {len(new_activation)} activations."
            )
        if exclude_self and len(new_prediction) != len(h_defense_prediction):
            raise ValueError("Self-exclusion requires aligned reference inputs.")

        ranks = []
        for sample_index, (prediction, activation) in enumerate(
            zip(new_prediction, new_activation)
        ):
            distances = torch.linalg.vector_norm(
                h_defense_activation - activation, dim=1
            )
            same_class = h_defense_prediction == prediction
            if exclude_self:
                same_class[sample_index] = False
            if not torch.any(same_class):
                raise ValueError(
                    f"No clean reference samples were predicted as class "
                    f"{prediction.item()}; cannot calculate rank for layer {layer}."
                )

            if distances[same_class].min() > tube_radius:
                ranks.append(len(h_defense_prediction))
                continue

            rank_distances = distances.clone()
            if exclude_self:
                rank_distances[sample_index] = torch.inf
            sorted_indices = torch.argsort(rank_distances)
            same_class_ranks = torch.where(
                h_defense_prediction[sorted_indices] == prediction
            )[0]
            ranks.append(int(same_class_ranks[0].item()))

        if layer not in layer_test_region_individual:
            layer_test_region_individual[layer] = {}
        layer_test_region_individual[layer][new_temp_label] = ranks
        return layer_test_region_individual

    def test(self):
        """
        The main testing procedure:
          1) Generate poison/clean sets.
          2) Create corresponding DataLoaders.
          3) Extract activations via hooks.
          4) Compute topological representations.
          5) (Optional) Visualization and outlier detection steps.
        """

        print('STEP 4')

        self.h_defense_ori_labels, self.h_defense_activations, self.h_defense_preds = (
            self.fetch_activation(self.reference_loader)
        )
        self.h_suspect_ori_labels, self.h_suspect_activations, self.h_suspect_preds = self.fetch_activation(
            self.suspect_loader)

        self.tube_radii = {}
        suspect_label = "Suspect"
        clean_label = "CleanValidation"
        for layer_ in self.h_defense_activations:
            tube_radius = self.getTubeRadius(
                self.h_defense_preds,
                self.h_defense_activations[layer_],
                layer_,
            )
            self.tube_radii[layer_] = tube_radius
            self.topological_representation = self.getLayerRegionDistance(
                new_prediction=self.h_defense_preds,
                new_activation=self.h_defense_activations[layer_],
                new_temp_label=clean_label,
                h_defense_prediction=self.h_defense_preds,
                h_defense_activation=self.h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=self.topological_representation,
                tube_radius=tube_radius,
                exclude_self=True,
            )
            self.topological_representation = self.getLayerRegionDistance(
                new_prediction=self.h_suspect_preds,
                new_activation=self.h_suspect_activations[layer_],
                new_temp_label=suspect_label,
                h_defense_prediction=self.h_defense_preds,
                h_defense_activation=self.h_defense_activations[layer_],
                layer=layer_,
                layer_test_region_individual=self.topological_representation,
                tube_radius=tube_radius,
            )

        expected_count = len(self.h_suspect_preds)
        layer_names = list(self.h_defense_activations)
        if not layer_names:
            raise RuntimeError("No activations were captured for TED++.")

        for layer in layer_names:
            layer_results = self.topological_representation[layer]
            if len(layer_results[suspect_label]) != expected_count:
                raise RuntimeError(
                    f"{layer}: expected {expected_count} suspect ranks, "
                    f"got {len(layer_results[suspect_label])}."
                )
            if len(layer_results[clean_label]) != len(self.h_defense_preds):
                raise RuntimeError(
                    f"{layer}: clean rank count does not match reference count."
                )

        clean_vectors = np.asarray(
            [
                [
                    self.topological_representation[layer][clean_label][index]
                    for layer in layer_names
                ]
                for index in range(len(self.h_defense_preds))
            ],
            dtype=np.float64,
        )
        suspect_vectors = np.asarray(
            [
                [
                    self.topological_representation[layer][suspect_label][index]
                    for layer in layer_names
                ]
                for index in range(expected_count)
            ],
            dtype=np.float64,
        )
        if expected_count == 0:
            raise ValueError("suspect_data must contain at least one input.")

        n_components = min(clean_vectors.shape[1] - 1, clean_vectors.shape[0] - 1)
        if n_components < 1:
            raise ValueError(
                "PCA reconstruction detection needs at least two clean rank vectors "
                "and two captured layers."
            )
        detector = PCA(contamination=self.contam, n_components=n_components)
        detector.fit(clean_vectors)
        scores = detector.decision_function(suspect_vectors)
        predictions = detector.predict(suspect_vectors)
        flagged = predictions == 1
        num_flagged = int(np.count_nonzero(flagged))
        flagged_fraction = num_flagged / expected_count

        return {
            "mode": "input-level",
            "method": "ted++",
            "verdict": (
                "likely backdoored"
                if flagged_fraction > self.contam
                else "likely clean"
            ),
            "num_inputs": expected_count,
            "num_flagged": num_flagged,
            "threshold": float(detector.threshold_),
            "parameters": {
                "reference_size": len(self.h_defense_preds),
                "samples_per_class": self.SAMPLES_PER_CLASS,
                "beta": self.beta,
                "contamination": self.contam,
                "batch_size": self.batch_size,
                "seed": self.seed,
            },
            "per_sample": [
                {
                    "index": index,
                    "predicted_label": int(self.h_suspect_preds[index].item()),
                    "score": float(score),
                    "flagged": bool(is_flagged),
                }
                for index, (score, is_flagged) in enumerate(zip(scores, flagged))
            ],
        }

    def detect(self):
        """
        Entry point for the detection procedure.
        """
        return self.test()

    def __del__(self):
        for h in self.hook_handles:
            h.remove()
        torch.cuda.empty_cache()


def tedpp(
    model,
    reference_data,
    suspect_data,
    contamination=0.01,
    batch_size=100,
    samples_per_class=5,
    beta=0.5,
    seed=None,
    device="cpu",
):
    detector = TEDPLUS(
        model,
        reference_data,
        suspect_data,
        contamination=contamination,
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        beta=beta,
        seed=seed,
        device=device,
    )
    try:
        return detector.detect()
    finally:
        for handle in detector.hook_handles:
            handle.remove()