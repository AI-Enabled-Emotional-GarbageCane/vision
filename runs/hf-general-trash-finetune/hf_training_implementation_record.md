# HF General Trash Fine-tune Implementation Record

Status: complete. Mainline training, ONNX export, evaluation, and threshold sweeps were observed.

Last observed: 2026-06-04 12:08:12 CST +0800

Recorder scope: this recorder did not start training or evaluation. It only observed local artifacts/processes and wrote this record.

## Data Sources

Final training dataset:

- `data/training/hf_general_trash_combined_yolo_cls`
- Manifest: `data/training/hf_general_trash_combined_yolo_cls/weak_dataset_manifest.csv`
- Summary: `data/training/hf_general_trash_combined_yolo_cls/dataset_summary.json`
- Copy mode: hardlink
- Total rows: 7142

Merged inputs:

| source key | local dataset |
| --- | --- |
| `realwaste` | `data/training/realwaste_yolo_cls_train_balanced` |
| `taco` | `data/training/taco_general_trash_hard_reject_yolo_cls` |
| `tidy` | `data/training/tidy_general_trash_yolo_cls` |
| `hf` | `data/training/hf_general_trash_sample_yolo_cls` |

HF subset details from the local trial report:

- `mnemoraorg/256x256-litter-sort-annotated-wastes`
  - Link: https://hf.co/datasets/mnemoraorg/256x256-litter-sort-annotated-wastes
  - Local source: `data/sources/hf_mnemora_256_litter_sort`
  - Local YOLO cls dataset: `data/training/hf_mnemora_litter_yolo_cls`
  - Mapping: `trash -> accept`; `cardboard`, `paper`, `metal`, `glass -> reject`; `plastic` ignored.
  - Materialized rows: 2044 after train reject balancing.
  - Issue noted by trial report: HF card text claims a larger balanced dataset, but the downloadable repo file tree/materialized local content did not match that full count.
- `1ease2/waste-garbage-management-dataset`
  - Link: https://hf.co/datasets/1ease2/waste-garbage-management-dataset
  - Local source sample: `data/sources/hf_1ease2_waste_sample`
  - Local YOLO cls dataset: `data/training/hf_1ease2_waste_sample_yolo_cls`
  - Mapping: `trash -> accept`; `battery`, `cardboard`, `glass`, `metal`, `paper`, `plastic -> reject`.
  - Downloaded sample: 250 per mapped class, 1750 raw images.
  - Materialized rows: 1050 after train reject balancing.
- `omasteam/waste-garbage-management-dataset`
  - Status: excluded duplicate candidate.
  - File-set check in trial report: same 19764 repo paths as `1ease2/waste-garbage-management-dataset`.

## Combined Dataset Counts

Split/label counts:

| split | accept | reject | total |
| --- | ---: | ---: | ---: |
| train | 1518 | 1472 | 2990 |
| val | 326 | 1752 | 2078 |
| test | 324 | 1750 | 2074 |
| total | 2168 | 4974 | 7142 |

Source/split/label counts:

| source | train accept | train reject | val accept | val reject | test accept | test reject | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hf` | 271 | 542 | 59 | 1083 | 57 | 1082 | 3094 |
| `realwaste` | 569 | 569 | 122 | 591 | 122 | 591 | 2564 |
| `taco` | 548 | 292 | 117 | 63 | 118 | 62 | 1200 |
| `tidy` | 130 | 69 | 28 | 15 | 27 | 15 | 284 |

HF-only merged subset counts:

| source | train accept | train reject | val accept | val reject | test accept | test reject | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mnemora` | 96 | 192 | 21 | 858 | 20 | 857 | 2044 |
| `1ease2` | 175 | 350 | 38 | 225 | 37 | 225 | 1050 |

## Pre-fine-tune HF Trial Evaluation

Observed local report: `runs/hf-general-trash-eval/hf_dataset_trial_report.md`

Evaluation target: HF-only merged test split from `data/training/hf_general_trash_sample_yolo_cls`, default accept gate threshold `0.76`.

| model | weak agreement | gate accept recall | reject false accept |
| --- | ---: | ---: | ---: |
| `realwaste-accuracy-002` | 93.42% | 1.75% | 1.48% |
| `general-trash-positive-002` | 72.43% | 5.26% | 2.96% |

Threshold sweep notes:

- `realwaste-accuracy-002` at threshold `0.50`: accept recall 7.02%, reject false accept 2.03%.
- `general-trash-positive-002` at threshold `0.50`: accept recall 42.11%, reject false accept 25.97%.

Interpretation recorded by trial report: these HF sources are useful as additional weak-label coverage, but the `trash` classes do not look like the current models' learned `accept` domain. Use as a controlled fine-tune candidate, not a threshold-only fix.

## Training Run

Observed running command from process table:

```bash
uv run --with ultralytics --with torch --with torchvision --with onnx --with onnxruntime --with onnxslim python scripts/train-yolo-cls.py --data data/training/hf_general_trash_combined_yolo_cls --model runs/general-trash-positive/general-trash-positive-002/weights/best.pt --project runs/hf-general-trash-finetune --serial-prefix hf-general-trash --epochs 25 --batch 32 --device 0 --workers 4 --patience 8 --export-onnx
```

Observed run directory:

- `runs/hf-general-trash-finetune/hf-general-trash-001`
- Run name: `hf-general-trash-001`
- `args.yaml` written.
- Training batch preview images observed: `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`.
- Final training artifacts observed:
  - `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.pt`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/weights/last.pt`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/results.csv`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/training_summary.json`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.onnx`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/confusion_matrix.png`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/confusion_matrix_normalized.png`
  - `runs/hf-general-trash-finetune/hf-general-trash-001/results.png`
  - validation preview images `val_batch*_labels.jpg` and `val_batch*_pred.jpg`

Training args observed in `args.yaml`:

| field | value |
| --- | --- |
| task | `classify` |
| mode | `train` |
| base model | `runs/general-trash-positive/general-trash-positive-002/weights/best.pt` |
| data | `data/training/hf_general_trash_combined_yolo_cls` |
| epochs | 25 |
| patience | 8 |
| batch | 32 |
| imgsz | 224 |
| device | `0` |
| workers | 4 |
| seed | 20260602 |
| deterministic | true |
| amp | true |
| project | `runs/hf-general-trash-finetune` |
| run name | `hf-general-trash-001` |

Base model artifacts observed:

- `runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- `runs/general-trash-positive/general-trash-positive-002/weights/last.pt`
- `runs/general-trash-positive/general-trash-positive-002/weights/best.onnx`

Expected fine-tune outputs from `scripts/train-yolo-cls.py` when training completes:

- `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.pt`
- `runs/hf-general-trash-finetune/hf-general-trash-001/weights/last.pt`
- ONNX export from the best checkpoint if present, otherwise last checkpoint.
- `runs/hf-general-trash-finetune/hf-general-trash-001/training_summary.json`

Training summary:

| field | value |
| --- | --- |
| completed epochs | 25 |
| image size | 224 |
| batch | 32 |
| device | `0` |
| CUDA available | true |
| CUDA device | `NVIDIA GeForce RTX 3060` |
| exported ONNX | `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.onnx` |

Training metrics from `results.csv`:

| metric | epoch | value |
| --- | ---: | ---: |
| best top1 accuracy | 17 | 0.87584 |
| best val/loss | 17 | 0.28546 |
| best train/loss | 24 | 0.27257 |
| final train/loss | 25 | 0.27915 |
| final top1 accuracy | 25 | 0.86766 |
| final top5 accuracy | 25 | 1.00000 |
| final val/loss | 25 | 0.33737 |

## Fine-tune Evaluation

Status: complete. Mainline evaluation produced summary, predictions, contact sheet, threshold sweep CSV, and threshold sweep summary for all observed targets. No active train/eval process was observed after completion.

Evaluated model:

- `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.onnx`

Observed evaluation shell:

```bash
MODEL="runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.onnx"
RUN="runs/hf-general-trash-finetune/hf-general-trash-001"
UV=(uv run --with onnxruntime --with pillow --with numpy python scripts/evaluate-weak-manifest.py)

"${UV[@]}" --dataset-dir data/training/hf_general_trash_sample_yolo_cls --manifest data/training/hf_general_trash_sample_yolo_cls/weak_dataset_manifest.csv --split test --model "$MODEL" --output "$RUN/hf_sample_test_predictions.csv" --summary "$RUN/hf_sample_test_summary.json" --contact-sheet "$RUN/hf_sample_test_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/training/hf_general_trash_combined_yolo_cls --manifest data/training/hf_general_trash_combined_yolo_cls/weak_dataset_manifest.csv --split test --model "$MODEL" --output "$RUN/hf_combined_test_predictions.csv" --summary "$RUN/hf_combined_test_summary.json" --contact-sheet "$RUN/hf_combined_test_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/inference_general_trash_positive/taco_full_accept_focus --manifest data/inference_general_trash_positive/taco_full_accept_focus/manifest.csv --model "$MODEL" --output "$RUN/taco_full_predictions.csv" --summary "$RUN/taco_full_summary.json" --contact-sheet "$RUN/taco_full_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/inference_extra_waste/taco_reject_safety --manifest data/inference_extra_waste/taco_reject_safety/manifest.csv --model "$MODEL" --output "$RUN/taco_reject_safety_predictions.csv" --summary "$RUN/taco_reject_safety_summary.json" --contact-sheet "$RUN/taco_reject_safety_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/training/tidy_general_trash_yolo_cls --manifest data/training/tidy_general_trash_yolo_cls/weak_dataset_manifest.csv --split test --model "$MODEL" --output "$RUN/tidy_test_predictions.csv" --summary "$RUN/tidy_test_summary.json" --contact-sheet "$RUN/tidy_test_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/training/realwaste_yolo_cls_full --manifest data/training/realwaste_yolo_cls_full/weak_dataset_manifest.csv --split test --model "$MODEL" --output "$RUN/realwaste_full_test_predictions.csv" --summary "$RUN/realwaste_full_test_summary.json" --contact-sheet "$RUN/realwaste_full_test_contact_sheet.jpg"
"${UV[@]}" --dataset-dir data/inference_taiwan_waste --manifest runs/realwaste-accuracy/realwaste-accuracy-002/weak_eval/taiwan_waste_mapped_manifest.csv --model "$MODEL" --output "$RUN/taiwan_mapped_predictions.csv" --summary "$RUN/taiwan_mapped_summary.json" --contact-sheet "$RUN/taiwan_mapped_contact_sheet.jpg"
```

Observed threshold sweep pattern:

```bash
python scripts/threshold-sweep.py --predictions "$RUN/<target>_predictions.csv" --output "$RUN/<target>_threshold_sweep.csv" --summary "$RUN/<target>_threshold_sweep_summary.json"
```

Completed eval artifacts:

| eval target | summary | predictions | contact sheet | threshold sweep |
| --- | --- | --- | --- | --- |
| HF-only sample test | `hf_sample_test_summary.json` | `hf_sample_test_predictions.csv` | `hf_sample_test_contact_sheet.jpg` | `hf_sample_test_threshold_sweep.csv`, `hf_sample_test_threshold_sweep_summary.json` |
| full combined test | `hf_combined_test_summary.json` | `hf_combined_test_predictions.csv` | `hf_combined_test_contact_sheet.jpg` | `hf_combined_test_threshold_sweep.csv`, `hf_combined_test_threshold_sweep_summary.json` |
| TACO full | `taco_full_summary.json` | `taco_full_predictions.csv` | `taco_full_contact_sheet.jpg` | `taco_full_threshold_sweep.csv`, `taco_full_threshold_sweep_summary.json` |
| TACO reject safety | `taco_reject_safety_summary.json` | `taco_reject_safety_predictions.csv` | `taco_reject_safety_contact_sheet.jpg` | `taco_reject_safety_threshold_sweep.csv`, `taco_reject_safety_threshold_sweep_summary.json` |
| TIDY test | `tidy_test_summary.json` | `tidy_test_predictions.csv` | `tidy_test_contact_sheet.jpg` | `tidy_test_threshold_sweep.csv`, `tidy_test_threshold_sweep_summary.json` |
| RealWaste full test | `realwaste_full_test_summary.json` | `realwaste_full_test_predictions.csv` | `realwaste_full_test_contact_sheet.jpg` | `realwaste_full_test_threshold_sweep.csv`, `realwaste_full_test_threshold_sweep_summary.json` |
| Taiwan mapped | `taiwan_mapped_summary.json` | `taiwan_mapped_predictions.csv` | `taiwan_mapped_contact_sheet.jpg` | `taiwan_mapped_threshold_sweep.csv`, `taiwan_mapped_threshold_sweep_summary.json` |

Important eval results at accept threshold `0.76`:

| target | count | weak agreement | gate accuracy | gate accept recall | reject false accept | accepted accept | false accepts on reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HF-only sample test | 1139 | 91.57% | 94.64% | 47.37% | 2.87% | 27 / 57 | 31 / 1082 |
| full combined test | 2074 | 88.62% | 89.87% | 58.95% | 4.40% | 191 / 324 | 77 / 1750 |
| TACO full | 1451 | 64.02% | 61.96% | 54.02% | 28.74% | 423 / 783 | 192 / 668 |
| TACO reject safety | 90 | 57.78% | 76.67% | 30.00% | 17.50% | 3 / 10 | 14 / 80 |
| TIDY test | 42 | 66.67% | 69.05% | 62.96% | 20.00% | 17 / 27 | 3 / 15 |
| RealWaste full test | 713 | 91.73% | 92.85% | 77.87% | 4.06% | 95 / 122 | 24 / 591 |
| Taiwan mapped | 50 | 54.00% | 50.00% | 4.00% | 4.00% | 1 / 25 | 1 / 25 |

Threshold sweep notes:

| target | threshold | gate accept recall | reject false accept | accepted accept | false accepts on reject |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF-only sample test | 0.95 | 29.82% | 0.65% | 17 / 57 | 7 / 1082 |
| full combined test | 0.95 | 34.26% | 0.97% | 111 / 324 | 17 / 1750 |
| RealWaste full test | 0.95 | 60.66% | 1.02% | 74 / 122 | 6 / 591 |
| TACO full | 0.95 | 8.17% | 3.29% | 64 / 783 | 22 / 668 |
| TACO reject safety | 0.95 | 10.00% | 0.00% | 1 / 10 | 0 / 80 |
| TIDY test | 0.95 | 40.74% | 13.33% | 11 / 27 | 2 / 15 |
| Taiwan mapped | 0.95 | 0.00% | 0.00% | 0 / 25 | 0 / 25 |

Comparison to pre-fine-tune HF-only trial:

- `general-trash-positive-002` on the HF-only merged test split: weak agreement 72.43%, gate accept recall 5.26%, reject false accept 2.96%.
- `hf-general-trash-001` on the same HF-only sample test split: weak agreement 91.57%, gate accept recall 47.37%, reject false accept 2.87%.

Interpretation notes:

- Fine-tune substantially improved HF-only sample accept recall versus `general-trash-positive-002` at the same default gate without increasing HF-only reject false accept rate.
- Full combined test and RealWaste full test remain usable at the default threshold, but reject false accept rises compared with HF-only sample.
- TACO full and TACO reject-safety evaluations show high false accept rates at the default threshold; raising the threshold reduces false accepts but sharply cuts accept recall.
- Taiwan mapped evaluation has very low accept recall at the default threshold and stays weak across the sweep.

## Issues / Caveats Observed

- The first evaluation shell appeared to stop after writing `hf_combined_test_summary.json` and `hf_combined_test_predictions.csv`; a later mainline retry completed `hf_combined_test_contact_sheet.jpg`, the remaining eval targets, and threshold sweeps.
- No standalone stdout/stderr evaluation log explaining the initial stop/retry was observed.
- No standalone stdout/stderr training log was observed inside `runs/hf-general-trash-finetune/hf-general-trash-001` at the initial observation time; only `args.yaml`, batch preview images, checkpoints, and `results.csv` were present.
- The mnemora HF dataset card describes a larger balanced dataset than the locally materialized/downloadable content used here.
- `omasteam/waste-garbage-management-dataset` was excluded because the local trial report found it duplicated the `1ease2` file set.
- All HF labels are weak labels. The local summaries explicitly warn to review label mappings before final training claims.
