# TODO - Diagnose Low Accuracy

## 1) Confirm evaluation mismatch (task-masking)

- [ ] Update `metrics/evaluator.py` to mask logits to the evaluated task’s class subset before `argmax`.

## 2) Confirm training is task-aware (masking during training)

- [ ] Optionally pass `TaskMask` into `BaseTrainer` from `train.py` (requires deciding intended metric/training protocol).

## 3) Re-run training and compare metrics

- [ ] Run `python train.py` and inspect final/average accuracy + accuracy matrix heatmap.

## 4) If still low, inspect loss scaling / replay targets

- [ ] Review `losses/losses.py` weights and `ReplayBuffer` behavior vs current training distribution.
- [ ] Review KD logits masking consistency between CE/KD/replay.
