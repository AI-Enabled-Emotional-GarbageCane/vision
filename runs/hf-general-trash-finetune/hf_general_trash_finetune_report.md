# HF General Trash Fine-tune Report

## Status

Result: **not a review candidate**. The run improves HF/RealWaste accept behavior, but it fails reject-safety gates on TACO/TIDY and still has weak Taiwan accept recall at the current gate.

No runtime default, AGX model, queue contract, or `recognition_result` payload was changed.

## Run

- Run dir: `runs/hf-general-trash-finetune/hf-general-trash-001`
- Base model: `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- Best PyTorch weights: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.pt`
- Exported ONNX: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.onnx`
- Serial prefix: `hf-general-trash`
- Device: `NVIDIA GeForce RTX 3060`
- Epochs requested: 25; early-stopping logic selected epoch 17 as best checkpoint.
- Best validation top1 from `results.csv`: 0.8758 at epoch 17; train loss 0.3429, val loss 0.2855.

Training command:

```bash
uv run --with ultralytics --with torch --with torchvision --with onnx --with onnxruntime --with onnxslim python scripts/train-yolo-cls.py --data data/training/hf_general_trash_combined_yolo_cls --model runs/general-trash-positive/general-trash-positive-002/weights/best.pt --project runs/hf-general-trash-finetune --serial-prefix hf-general-trash --epochs 25 --batch 32 --device 0 --workers 4 --patience 8 --export-onnx
```

## Dataset

HF trial datasets:

- `mnemoraorg/256x256-litter-sort-annotated-wastes`: mapped `trash` to accept; `cardboard`, `paper`, `metal`, `glass` to reject; `plastic` ignored.
- `1ease2/waste-garbage-management-dataset`: sampled 250 per mapped class; `trash` to accept; `battery`, `cardboard`, `glass`, `metal`, `paper`, `plastic` to reject.
- `omasteam/waste-garbage-management-dataset`: file set matched `1ease2`, treated as duplicate and not merged.

HF sample dataset counts:

| Split/class | Count |
|---|---:|
| `train/accept` | 271 |
| `train/reject` | 542 |
| `val/accept` | 59 |
| `val/reject` | 1083 |
| `test/accept` | 57 |
| `test/reject` | 1082 |
| total | 3094 |

Combined fine-tune dataset counts:

| Split/class | Count |
|---|---:|
| `train/accept` | 1518 |
| `train/reject` | 1472 |
| `val/accept` | 326 |
| `val/reject` | 1752 |
| `test/accept` | 324 |
| `test/reject` | 1750 |
| total | 7142 |

Combined sources: `realwaste_yolo_cls_train_balanced`, `taco_general_trash_hard_reject_yolo_cls`, `tidy_general_trash_yolo_cls`, `hf_general_trash_sample_yolo_cls`.

## Gate Metrics At Runtime Threshold

Runtime gate used for evaluation: `accept_threshold=0.76`, `uncertain_threshold=0.50`.

| Evaluation set | Count | Weak agreement | Accept recall | Reject false accept | Result |
|---|---:|---:|---:|---:|---|
| HF sample test | 1139 | 91.6% | 47.4% | 2.9% | pass safety |
| HF combined test | 2074 | 88.6% | 59.0% | 4.4% | pass safety |
| TACO full accept-focus | 1451 | 64.0% | 54.0% | 28.7% | fail safety |
| TACO reject-safety | 90 | 57.8% | 30.0% | 17.5% | fail safety |
| TIDY test | 42 | 66.7% | 63.0% | 20.0% | fail safety |
| RealWaste full test | 713 | 91.7% | 77.9% | 4.1% | pass safety |
| Taiwan mapped weak set | 50 | 54.0% | 4.0% | 4.0% | pass safety |

## Threshold Sweep

| Evaluation set | Best recall with reject FAR <= 10% | Max recall in sweep |
|---|---|---|
| HF sample test | t=0.50, recall=73.7%, FAR=7.5% | t=0.50, recall=73.7%, FAR=7.5% |
| HF combined test | t=0.50, recall=79.0%, FAR=9.6% | t=0.50, recall=79.0%, FAR=9.6% |
| TACO full accept-focus | t=0.95, recall=8.2%, FAR=3.3% | t=0.50, recall=89.1%, FAR=65.4% |
| TACO reject-safety | t=0.85, recall=30.0%, FAR=7.5% | t=0.50, recall=70.0%, FAR=43.8% |
| TIDY test | none | t=0.50, recall=74.1%, FAR=46.7% |
| RealWaste full test | t=0.50, recall=84.4%, FAR=6.8% | t=0.50, recall=84.4%, FAR=6.8% |
| Taiwan mapped weak set | t=0.75, recall=4.0%, FAR=4.0% | t=0.50, recall=28.0%, FAR=20.0% |

## Interpretation

- The HF-added fine-tune materially improves the newly collected HF sample accept recall versus the previous `general-trash-positive-002` baseline, but only to 47.4% at the current runtime gate and 73.7% at threshold 0.50.
- RealWaste full test improves to 77.9% accept recall with 4.1% reject false accept at the current gate.
- The safety regression is too large on TACO: 28.7% false accept on TACO full reject labels and 17.5% on the dedicated TACO reject-safety set.
- The Taiwan mapped weak set remains poor for accept behavior: 4.0% accept recall at the current gate, suggesting dataset/domain mismatch rather than only a threshold issue.

## Subagent Review

The metrics-review subagent independently confirmed epoch 17 as best, noted mild-to-moderate overfitting after that point, and flagged the same reject-safety regression on TACO/TIDY. It recommended keeping this as an experiment rather than replacing the broad reject gate at threshold 0.76.

## Recommendation

Do not deploy or mark this run as a review candidate. Keep it as an experiment showing that HF general-trash positives help recall but introduce reject-safety risk. The next useful direction is to add L515/Taiwan real accept positives and hard reject negatives, or train a two-stage crop/detection pipeline so small wrappers, tissue, cigarette butts, garbage bags, and rigid recyclables are learned from similar camera geometry.

## Artifacts

- HF sample test: `runs/hf-general-trash-finetune/hf-general-trash-001/hf_sample_test_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_sample_test_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_sample_test_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_sample_test_threshold_sweep.csv`
- HF combined test: `runs/hf-general-trash-finetune/hf-general-trash-001/hf_combined_test_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_combined_test_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_combined_test_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/hf_combined_test_threshold_sweep.csv`
- TACO full accept-focus: `runs/hf-general-trash-finetune/hf-general-trash-001/taco_full_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_full_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_full_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_full_threshold_sweep.csv`
- TACO reject-safety: `runs/hf-general-trash-finetune/hf-general-trash-001/taco_reject_safety_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_reject_safety_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_reject_safety_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/taco_reject_safety_threshold_sweep.csv`
- TIDY test: `runs/hf-general-trash-finetune/hf-general-trash-001/tidy_test_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/tidy_test_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/tidy_test_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/tidy_test_threshold_sweep.csv`
- RealWaste full test: `runs/hf-general-trash-finetune/hf-general-trash-001/realwaste_full_test_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/realwaste_full_test_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/realwaste_full_test_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/realwaste_full_test_threshold_sweep.csv`
- Taiwan mapped weak set: `runs/hf-general-trash-finetune/hf-general-trash-001/taiwan_mapped_summary.json`, `runs/hf-general-trash-finetune/hf-general-trash-001/taiwan_mapped_predictions.csv`, `runs/hf-general-trash-finetune/hf-general-trash-001/taiwan_mapped_contact_sheet.jpg`, `runs/hf-general-trash-finetune/hf-general-trash-001/taiwan_mapped_threshold_sweep.csv`
