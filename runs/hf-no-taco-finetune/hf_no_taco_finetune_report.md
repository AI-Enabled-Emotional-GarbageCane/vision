# HF No-TACO Fine-tune Report

## Status

Result: **not a review candidate**. Removing TACO from training improved the no-TACO internal holdout and slightly improved RealWaste, but it worsened TACO/TIDY reject safety and did not improve Taiwan accept behavior.

No runtime default, AGX model, queue contract, or `recognition_result` payload was changed.

## Run

- Run dir: `runs/hf-no-taco-finetune/hf-no-taco-001`
- Base model: `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- Best PyTorch weights: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.pt`
- Exported ONNX: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.onnx`
- Serial prefix: `hf-no-taco`
- Device: `NVIDIA GeForce RTX 3060`
- Best validation top1: 0.9257 at epoch 19; train loss 0.2333, val loss 0.2006.
- Final epoch top1: 0.9167; train loss 0.1599, val loss 0.2367.

Training command:

```bash
uv run --with ultralytics --with torch --with torchvision --with onnx --with onnxruntime --with onnxslim python scripts/train-yolo-cls.py --data data/training/hf_no_taco_combined_yolo_cls --model runs/general-trash-positive/general-trash-positive-002/weights/best.pt --project runs/hf-no-taco-finetune --serial-prefix hf-no-taco --epochs 25 --batch 32 --device 0 --workers 4 --patience 8 --export-onnx
```

## Dataset

Merged sources: `realwaste_yolo_cls_train_balanced`, `tidy_general_trash_yolo_cls`, `hf_general_trash_sample_yolo_cls`. TACO was excluded from training for this experiment.

| Split/class | Count |
|---|---:|
| `train/accept` | 970 |
| `train/reject` | 1180 |
| `val/accept` | 209 |
| `val/reject` | 1689 |
| `test/accept` | 206 |
| `test/reject` | 1688 |
| total | 5942 |

## Gate Metrics At Runtime Threshold

Runtime gate used for evaluation: `accept_threshold=0.76`, `uncertain_threshold=0.50`.

| Evaluation set | Count | Weak agreement | Accept recall | Reject false accept | Result |
|---|---:|---:|---:|---:|---|
| HF sample test | 1139 | 93.0% | 49.1% | 2.5% | pass safety |
| HF combined test | 2074 | 88.9% | 56.8% | 4.3% | pass safety |
| HF no-TACO combined test | 1894 | 92.4% | 70.9% | 3.2% | pass safety |
| TACO full accept-focus | 1451 | 50.7% | 32.6% | 33.2% | fail safety |
| TACO reject-safety | 90 | 67.8% | 40.0% | 21.2% | fail safety |
| TIDY test | 42 | 71.4% | 66.7% | 26.7% | fail safety |
| RealWaste full test | 713 | 92.7% | 82.0% | 3.9% | pass safety |
| Taiwan mapped weak set | 50 | 52.0% | 0.0% | 0.0% | pass safety |

## Comparison To HF+TACO Previous Run

Previous run: `runs/hf-general-trash-finetune/hf-general-trash-001` at the same gate.

| Evaluation set | Previous recall | No-TACO recall | Previous reject FAR | No-TACO reject FAR |
|---|---:|---:|---:|---:|
| HF sample test | 47.4% | 49.1% | 2.9% | 2.5% |
| HF combined test | 59.0% | 56.8% | 4.4% | 4.3% |
| TACO full accept-focus | 54.0% | 32.6% | 28.7% | 33.2% |
| TACO reject-safety | 30.0% | 40.0% | 17.5% | 21.2% |
| TIDY test | 63.0% | 66.7% | 20.0% | 26.7% |
| RealWaste full test | 77.9% | 82.0% | 4.1% | 3.9% |
| Taiwan mapped weak set | 4.0% | 0.0% | 4.0% | 0.0% |

## Threshold Sweep

| Evaluation set | Best recall with reject FAR <= 10% | Max recall in sweep |
|---|---|---|
| HF sample test | t=0.50, recall=66.7%, FAR=5.6% | t=0.50, recall=66.7%, FAR=5.6% |
| HF combined test | t=0.50, recall=68.2%, FAR=7.3% | t=0.50, recall=68.2%, FAR=7.3% |
| HF no-TACO combined test | t=0.50, recall=78.6%, FAR=5.9% | t=0.50, recall=78.6%, FAR=5.9% |
| TACO full accept-focus | none | t=0.50, recall=52.6%, FAR=51.5% |
| TACO reject-safety | t=0.90, recall=30.0%, FAR=10.0% | t=0.50, recall=40.0%, FAR=28.7% |
| TIDY test | none | t=0.50, recall=74.1%, FAR=33.3% |
| RealWaste full test | t=0.50, recall=85.2%, FAR=5.8% | t=0.50, recall=85.2%, FAR=5.8% |
| Taiwan mapped weak set | t=0.50, recall=8.0%, FAR=4.0% | t=0.50, recall=8.0%, FAR=4.0% |

## Interpretation

- No-TACO improves the internal no-TACO holdout, but that is not enough evidence for deployment because the external TACO/TIDY reject-safety sets get worse.
- Compared with the previous HF+TACO run, TACO full false accept increases from 28.7% to 33.2%, and TACO reject-safety increases from 17.5% to 21.3%.
- Taiwan mapped accept recall drops from 4.0% to 0.0% at the current gate, so this does not address the real-scene issue.
- Threshold sweep cannot rescue TACO full safety; no tested threshold reaches reject FAR <= 10%.

## Recommendation

Do not continue by simply removing TACO. The next better approach is not HF-only retraining, but targeted data repair: add L515/Taiwan accept positives for soft wrappers/tissue/cigarette/garbage bag/small misc, plus hard reject negatives for bottles/cans/caps/paper/glass/metal under similar camera geometry. Then train with balanced hard negatives or move to a crop/detection pipeline.

## Artifacts

- HF sample test: `runs/hf-no-taco-finetune/hf-no-taco-001/hf_sample_test_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_sample_test_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_sample_test_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_sample_test_contact_sheet.jpg`
- HF combined test: `runs/hf-no-taco-finetune/hf-no-taco-001/hf_combined_test_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_combined_test_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_combined_test_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_combined_test_contact_sheet.jpg`
- HF no-TACO combined test: `runs/hf-no-taco-finetune/hf-no-taco-001/hf_no_taco_combined_test_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_no_taco_combined_test_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_no_taco_combined_test_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/hf_no_taco_combined_test_contact_sheet.jpg`
- TACO full accept-focus: `runs/hf-no-taco-finetune/hf-no-taco-001/taco_full_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_full_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_full_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_full_contact_sheet.jpg`
- TACO reject-safety: `runs/hf-no-taco-finetune/hf-no-taco-001/taco_reject_safety_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_reject_safety_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_reject_safety_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taco_reject_safety_contact_sheet.jpg`
- TIDY test: `runs/hf-no-taco-finetune/hf-no-taco-001/tidy_test_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/tidy_test_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/tidy_test_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/tidy_test_contact_sheet.jpg`
- RealWaste full test: `runs/hf-no-taco-finetune/hf-no-taco-001/realwaste_full_test_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/realwaste_full_test_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/realwaste_full_test_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/realwaste_full_test_contact_sheet.jpg`
- Taiwan mapped weak set: `runs/hf-no-taco-finetune/hf-no-taco-001/taiwan_mapped_summary.json`, `runs/hf-no-taco-finetune/hf-no-taco-001/taiwan_mapped_predictions.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taiwan_mapped_threshold_sweep.csv`, `runs/hf-no-taco-finetune/hf-no-taco-001/taiwan_mapped_contact_sheet.jpg`
