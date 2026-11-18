import json
import tempfile
import unittest
from pathlib import Path

from scripts.coach_core.commands import training_command
from scripts.coach_core.configuration import load_config
from scripts.preflight_checks.artifacts import check_configuration

try:
    import torch
except ImportError:
    torch = None


class ObjectiveConfigurationTests(unittest.TestCase):
    def test_config_sets_the_objective_for_the_shard_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            config_file = run / "config.json"
            config_file.write_text(
                json.dumps(
                    {
                        "training_objective": "dpace",
                        "initial_draft": "draft.gguf",
                        "target_model": "target.gguf",
                    }
                )
            )
            config = load_config(config_file)
            plan = {"config": config, "steps_per_epoch_estimate": 1}
            (run / "plan.json").write_text(json.dumps(plan))
            command = training_command(config, plan, run, run, {"key": "shard"}, 0, 1e-5)
            self.assertEqual(command[command.index("--objective") + 1], "dpace")
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                check_configuration({**config, "training_objective": "uniform"}, plan)

    def test_old_config_and_plan_default_to_uniform(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_file = Path(temporary) / "config.json"
            config_file.write_text("{}")
            config = load_config(config_file)
            self.assertEqual(config["training_objective"], "uniform")
            old_config = dict(config)
            del old_config["training_objective"]
            check_configuration(config, {"config": old_config})


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PositionLossTests(unittest.TestCase):
    def test_confident_prefix_weights_are_detached_and_normalized_by_blocks(self):
        from scripts.training.position_loss import dpace_loss

        token_nll = torch.zeros(2, 3, requires_grad=True)
        selector_nll = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        result = dpace_loss(token_nll, selector_nll)
        torch.testing.assert_close(result, torch.tensor(1.75))
        result.backward()

        expected_gradient = torch.tensor([[1.5, 1.0, 0.5], [1.5, 1.0, 0.5]])
        torch.testing.assert_close(token_nll.grad, expected_gradient)
