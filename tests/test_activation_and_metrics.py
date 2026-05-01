import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn

from calibract.experiment_logging import compute_ece
from calibract.models.activation_configs import activations
from calibract.models.activations import ChannelwiseActivation, KGActivationLaplacian
from calibract.training.loops import build_activation_map


class ActivationAndMetricsTests(unittest.TestCase):
    def test_ece_is_zero_for_perfect_confidence(self):
        probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])

        ece = compute_ece(probs, targets, n_bins=5)

        self.assertAlmostEqual(float(ece), 0.0, places=6)

    def test_channelwise_kglap_map_targets_stage4_act2(self):
        activation_map = build_activation_map(activations["stage4_act2only_channelwise_kglap"])

        self.assertIsInstance(activation_map["layer4.0.act2"], ChannelwiseActivation)
        self.assertEqual(len(activation_map["layer4.0.act2"].activations), 512)
        self.assertIsInstance(activation_map["layer4.0.act2"].activations[0], KGActivationLaplacian)
        self.assertIsInstance(activation_map["layer3.0.act2"], nn.ReLU)


if __name__ == "__main__":
    unittest.main()
