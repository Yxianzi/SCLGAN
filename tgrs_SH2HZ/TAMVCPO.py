import torch.nn.functional as F
import torch
from collections import defaultdict
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
import numpy as np


class MVCPO_Helper:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.memory_bank = {c: {'features': [], 'probs': []} for c in range(num_classes)}
        self.view_weights = torch.ones(3) / 3
        self.topo_diff_history = []

    def select_pseudo_labels(self, pred_v0, pred_v1, pred_v2, epoch, topo_diff=None):
        if topo_diff is not None:
            self.topo_diff_history.append(topo_diff)
            if len(self.topo_diff_history) > 100:
                self.topo_diff_history.pop(0)
        topo_diff = np.mean(self.topo_diff_history) if self.topo_diff_history else None
        conf_thresh = self._get_dynamic_threshold(epoch, topo_diff)

        prob_v0, label_v0 = torch.max(F.softmax(pred_v0, dim=1), dim=1)
        prob_v1, label_v1 = torch.max(F.softmax(pred_v1, dim=1), dim=1)
        prob_v2, label_v2 = torch.max(F.softmax(pred_v2, dim=1), dim=1)

        consistent_mask = (label_v0 == label_v1) & (label_v0 == label_v2)
        consistent_mask &= (prob_v0 > conf_thresh) & (prob_v1 > conf_thresh) & (prob_v2 > conf_thresh)

        pair_conf_thresh = conf_thresh * 1.2
        pair_mask_v0v1 = (label_v0 == label_v1) & ((prob_v0 + prob_v1) > pair_conf_thresh)
        pair_mask_v0v2 = (label_v0 == label_v2) & ((prob_v0 + prob_v2) > pair_conf_thresh)
        pair_mask_v1v2 = (label_v1 == label_v2) & ((prob_v1 + prob_v2) > pair_conf_thresh)

        final_mask = consistent_mask | pair_mask_v0v1 | pair_mask_v0v2 | pair_mask_v1v2
        pseudo_labels = label_v0.clone()
        pseudo_labels[pair_mask_v0v1] = label_v0[pair_mask_v0v1]
        pseudo_labels[pair_mask_v0v2] = label_v0[pair_mask_v0v2]
        pseudo_labels[pair_mask_v1v2] = label_v1[pair_mask_v1v2]

        if len(pseudo_labels) != pred_v0.shape[0]:
            print(f" {len(pseudo_labels)} != {pred_v0.shape[0]}")
            final_mask = torch.ones(pred_v0.shape[0], dtype=torch.bool, device=pred_v0.device)
            return pred_v0.argmax(dim=1), final_mask, pred_v0

        return pseudo_labels[final_mask], final_mask, pred_v0[final_mask]

    def _get_dynamic_threshold(self, epoch, topo_diff=None):
        base_thresh = 0.9
        min_thresh = 0.7
        progress = min(epoch / 100, 1.0)
        thres = base_thresh - (base_thresh - min_thresh) * progress  # 基础衰减
        if topo_diff is not None:
            thres += 0.3 * topo_diff  # 在衰减基础上调整
            thres = min(base_thresh, thres)

        return max(min_thresh, thres)

class TopologyValidator:
    def __init__(self, n_classes, homology_dimensions=(0, 1)):
        self.n_classes = n_classes
        self.vrp = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions,
            collapse_edges=True
        )
        self.amplitude = Amplitude(metric='wasserstein')
        self.class_diffs_history = defaultdict(list)
        self.validation_stats = {
            'total_checked': 0,
            'filtered_classes': defaultdict(int),
            'corrected_points': 0,
            'invalid_diagrams': 0
        }

    def compute_topology_diff(self, source_feat, source_label, target_feat, target_label):

        source_feat = source_feat.detach().cpu().numpy()
        source_label = source_label.detach().cpu().numpy().flatten()
        target_feat = target_feat.detach().cpu().numpy()
        target_label = target_label.detach().cpu().numpy().flatten()

        diff_scores = {}
        for c in range(self.n_classes):
            try:
                src_points = source_feat[source_label == c] if len(source_feat) > 0 else None
                tgt_points = target_feat[target_label == c] if len(target_feat) > 0 else None

                if src_points is None or tgt_points is None or len(src_points) < 2 or len(tgt_points) < 2:
                    continue
                src_diagrams = self._correct_invalid_points(self.vrp.fit_transform(src_points[None, :]))
                tgt_diagrams = self._correct_invalid_points(self.vrp.fit_transform(tgt_points[None, :]))

                if src_diagrams is None or tgt_diagrams is None or src_diagrams.shape[1] == 0 or tgt_diagrams.shape[
                    1] == 0:
                    continue
                min_points = min(src_diagrams.shape[1], tgt_diagrams.shape[1])
                if min_points == 0:
                    continue
                src_diagrams = src_diagrams[:, :min_points, :]
                tgt_diagrams = tgt_diagrams[:, :min_points, :]
                try:
                    diff = self.amplitude.fit_transform(np.abs(src_diagrams - tgt_diagrams))
                    diff_scores[c] = diff.mean()
                except:
                    continue
            except Exception as e:
                print(f"Class {c}拓扑计算异常: {str(e)}")
                continue
        return diff_scores

    def _correct_invalid_points(self, diagrams):
        self.validation_stats['corrected_points'] += 1
        if diagrams is None or diagrams.shape[1] == 0:
            return None
        if len(diagrams.shape) != 3:
            diagrams = diagrams.reshape(1, -1, 2)
        diagrams = np.nan_to_num(diagrams, nan=0.0, posinf=1e6, neginf=-1e6)
        invalid_mask = diagrams[0, :, 1] < diagrams[0, :, 0]
        if invalid_mask.any():
            print(f" {invalid_mask.sum()} ")
            diagrams[0, invalid_mask, 1] = diagrams[0, invalid_mask, 0] + 1e-5
        assert np.all(diagrams[0, :, 1] >= diagrams[0, :, 0]), "."
        return diagrams

    def validate_pseudo_labels(self, source_feat, source_label, target_feat, pseudo_labels,epoch):
        diff_scores = self.compute_topology_diff(source_feat, source_label, target_feat, pseudo_labels)
        avg_diff = np.mean(list(diff_scores.values())) if diff_scores else 0.0
        if not diff_scores:
            return torch.zeros_like(pseudo_labels, dtype=torch.bool), 0.0

        invalid_mask = torch.zeros_like(pseudo_labels, dtype=torch.bool)
        return invalid_mask, avg_diff
