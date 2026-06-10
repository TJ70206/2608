import unittest

import numpy as np
import torch

from src.xa202608.losses import reliability_weighted_stage_prototype_alignment_loss
from scripts.train_transfer import (
    _apply_time_aware_output_calibration,
    _batch_time_order_consistency_losses,
    _fit_time_aware_output_calibration,
    _prototype_alignment_loss,
    _resolve_transfer_steps,
    _scheduled_alignment_weight,
    _sequence_temporal_consistency_losses,
    _stage_auxiliary_loss,
    _supervised_loss,
    _target_stage_aux_uses_unlabeled_stages,
)


class _SizedLoader:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length


class TrainTransferOptionTests(unittest.TestCase):
    def test_resolve_transfer_steps_defaults_to_source_loader_length(self) -> None:
        self.assertEqual(_resolve_transfer_steps({}, _SizedLoader(8), _SizedLoader(200)), 8)

    def test_resolve_transfer_steps_can_use_target_loader_length(self) -> None:
        self.assertEqual(_resolve_transfer_steps({"epoch_steps": "target"}, _SizedLoader(8), _SizedLoader(200)), 200)

    def test_resolve_transfer_steps_explicit_value_takes_precedence(self) -> None:
        self.assertEqual(_resolve_transfer_steps({"epoch_steps": "target", "steps_per_epoch": 16}, _SizedLoader(8), _SizedLoader(200)), 16)

    def test_alignment_warmup_reaches_base_weight(self) -> None:
        self.assertAlmostEqual(_scheduled_alignment_weight(0.003, epoch=5, warmup_epochs=10), 0.0015)
        self.assertAlmostEqual(_scheduled_alignment_weight(0.003, epoch=10, warmup_epochs=10), 0.003)

    def test_supervised_loss_can_upweight_late_stage_samples(self) -> None:
        criterion = torch.nn.SmoothL1Loss(reduction="none")
        pred = torch.tensor([[0.0], [0.0]])
        target = torch.tensor([[0.4], [0.4]])
        stage = torch.tensor([0, 2])

        plain = _supervised_loss(criterion, pred, target, stage)
        weighted = _supervised_loss(criterion, pred, target, stage, late_stage_weight=3.0)

        self.assertGreater(float(weighted), float(plain))

    def test_sequence_temporal_consistency_penalizes_rul_increases_only(self) -> None:
        decreasing = torch.tensor([[[0.8], [0.6], [0.4]], [[0.5], [0.5], [0.3]]])
        increasing = torch.tensor([[[0.4], [0.5], [0.6]], [[0.3], [0.2], [0.4]]])

        dec_mono, dec_smooth = _sequence_temporal_consistency_losses(decreasing)
        inc_mono, inc_smooth = _sequence_temporal_consistency_losses(increasing)

        self.assertAlmostEqual(float(dec_mono), 0.0)
        self.assertGreater(float(inc_mono), float(dec_mono))
        self.assertGreater(float(dec_smooth), 0.0)
        self.assertGreater(float(inc_smooth), 0.0)

    def test_sequence_temporal_consistency_is_differentiable(self) -> None:
        pred = torch.tensor([[[0.4], [0.5], [0.45]]], requires_grad=True)

        mono, smooth = _sequence_temporal_consistency_losses(pred)
        (mono + smooth).backward()

        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, pred.shape)

    def test_batch_time_order_consistency_penalizes_same_unit_lifecycle_increases(self) -> None:
        unit_id = torch.tensor([2, 1, 1, 2, 1])
        time_index = torch.tensor([10, 20, 10, 20, 30])
        good_pred = torch.tensor([[0.5], [0.6], [0.8], [0.4], [0.4]])
        bad_pred = torch.tensor([[0.5], [0.9], [0.6], [0.7], [0.8]])

        good_mono, good_smooth = _batch_time_order_consistency_losses(good_pred, unit_id, time_index)
        bad_mono, bad_smooth = _batch_time_order_consistency_losses(bad_pred, unit_id, time_index)

        self.assertAlmostEqual(float(good_mono), 0.0)
        self.assertGreater(float(bad_mono), float(good_mono))
        self.assertGreaterEqual(float(good_smooth), 0.0)
        self.assertGreaterEqual(float(bad_smooth), 0.0)

    def test_batch_time_order_consistency_is_differentiable(self) -> None:
        pred = torch.tensor([[0.4], [0.5], [0.45]], requires_grad=True)
        unit_id = torch.tensor([1, 1, 1])
        time_index = torch.tensor([1, 2, 3])

        mono, smooth = _batch_time_order_consistency_losses(pred, unit_id, time_index)
        (mono + smooth).backward()

        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.shape, pred.shape)

    def test_stage_auxiliary_loss_prefers_correct_stage_logits(self) -> None:
        stage = torch.tensor([0, 1, 2])
        good_logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        bad_logits = torch.tensor([[0.0, 0.0, 4.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

        good_loss = _stage_auxiliary_loss(good_logits, stage)
        bad_loss = _stage_auxiliary_loss(bad_logits, stage)

        self.assertLess(float(good_loss), float(bad_loss))

    def test_target_stage_auxiliary_allows_only_unlabeled_pseudo_time_stages(self) -> None:
        self.assertTrue(_target_stage_aux_uses_unlabeled_stages({"target_stage_source": "time_progress"}))
        self.assertTrue(_target_stage_aux_uses_unlabeled_stages({"target_stage_source": "pseudo_time"}))
        self.assertFalse(_target_stage_aux_uses_unlabeled_stages({"target_stage_source": "rul"}))
        self.assertFalse(_target_stage_aux_uses_unlabeled_stages({}))

    def test_rspa_loss_is_lower_for_matched_stage_prototypes(self) -> None:
        source = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [3.2, 0.0]])
        target_good = torch.tensor([[0.1, 0.0], [0.3, 0.0], [3.1, 0.0], [3.3, 0.0]])
        target_bad = torch.tensor([[3.1, 0.0], [3.3, 0.0], [0.1, 0.0], [0.3, 0.0]])
        stages = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])

        good = reliability_weighted_stage_prototype_alignment_loss(source, target_good, stages, stages, logits, num_stages=2)
        bad = reliability_weighted_stage_prototype_alignment_loss(source, target_bad, stages, stages, logits, num_stages=2)

        self.assertLess(float(good), float(bad))

    def test_rspa_loss_downweights_uncertain_target_stage_predictions(self) -> None:
        source = torch.tensor([[0.0, 0.0], [0.2, 0.0], [3.0, 0.0], [3.2, 0.0]])
        target_bad = torch.tensor([[3.1, 0.0], [3.3, 0.0], [0.1, 0.0], [0.3, 0.0]])
        stages = torch.tensor([0, 0, 1, 1])
        confident = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
        uncertain = torch.zeros_like(confident)

        confident_loss = reliability_weighted_stage_prototype_alignment_loss(
            source, target_bad, stages, stages, confident, num_stages=2
        )
        uncertain_loss = reliability_weighted_stage_prototype_alignment_loss(
            source, target_bad, stages, stages, uncertain, num_stages=2
        )

        self.assertLess(float(uncertain_loss), float(confident_loss))

    def test_prototype_alignment_helper_requires_target_stage_logits_when_enabled(self) -> None:
        source = torch.randn(4, 2)
        target = torch.randn(4, 2)
        stages = torch.tensor([0, 0, 1, 1])

        with self.assertRaises(ValueError):
            _prototype_alignment_loss(
                source,
                target,
                stages,
                stages,
                target_stage_logits=None,
                num_stages=2,
                min_confidence=0.0,
                enabled=True,
            )

    def test_time_aware_output_calibration_uses_validation_mapping(self) -> None:
        val_pred = {
            "y_pred": np.asarray([0.20, 0.40, 0.60, 0.80], dtype=np.float64),
            "y_true": np.asarray([0.30, 0.45, 0.55, 0.70], dtype=np.float64),
            "time_index": np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        }
        calibration = _fit_time_aware_output_calibration(val_pred, {"degree": 1, "ridge": 0.0})
        raw_rmse = float(np.sqrt(np.mean((val_pred["y_true"] - val_pred["y_pred"]) ** 2)))

        calibrated = {key: value.copy() for key, value in val_pred.items()}
        _apply_time_aware_output_calibration(calibrated, calibration, clip_range=(0.0, 1.0))
        calibrated_rmse = float(np.sqrt(np.mean((calibrated["y_true"] - calibrated["y_pred"]) ** 2)))

        self.assertLess(calibrated_rmse, raw_rmse)
        self.assertIn("coef", calibration)
        self.assertEqual(int(calibration["degree"]), 1)

    def test_time_aware_output_calibration_clips_outputs(self) -> None:
        val_pred = {
            "y_pred": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            "y_true": np.asarray([0.0, 1.0, 1.0], dtype=np.float64),
            "time_index": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        }
        calibration = {"degree": 1, "ridge": 0.0, "time_min": 0.0, "time_scale": 2.0, "coef": np.asarray([2.0, 1.0, 0.0])}

        pred = {key: value.copy() for key, value in val_pred.items()}
        _apply_time_aware_output_calibration(pred, calibration, clip_range=(0.0, 1.0))

        self.assertTrue(np.all(pred["y_pred"] >= 0.0))
        self.assertTrue(np.all(pred["y_pred"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
