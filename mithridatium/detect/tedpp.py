import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from pyod.models.pca import PCA
from torchmetrics.functional import pairwise_euclidean_distance


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

        # Set up hooks for activation extraction
        self.register_hooks()

        self.topological_representation = {}

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


    # ==============================
    #       HOOK & MAIN TEST
    # ==============================
    def fetch_activation(self, loader):
        self.model.eval()

        all_h_label, pred_set = [], []
        activation_container = {}

        for images, labels in loader:
            _ = self.model(images.to(self.device))
            break
        for key in self.activations:
            activation_container[key] = []
        self.activations.clear()

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            output = self.model(images.to(self.device))

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

        for key in activation_container:
            activation_container[key] = torch.stack(activation_container[key])
        all_h_label = torch.stack(all_h_label)
        pred_set = torch.cat(pred_set)

        return all_h_label, activation_container, pred_set

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
