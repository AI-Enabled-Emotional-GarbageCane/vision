# HF no-TACO fine-tune experiment record

Last updated: 2026-06-04 12:31:29 CST (+0800)

## Scope

- Role: no-TACO fine-tune experiment record sub-agent.
- Write scope: this file only.
- Training status: training completed; no training was started by this record agent.
- Project directory: `runs/hf-no-taco-finetune`
- Serial prefix: `hf-no-taco`
- Active run observed: `hf-no-taco-001`

## Dataset

- Dataset path: `data/training/hf_no_taco_combined_yolo_cls`
- Dataset format: YOLO classification with `accept` and `reject` classes.
- Copy mode recorded by dataset summary: hardlink.
- Summary note from dataset artifact: merged weak-label YOLO classification dataset; labels should be reviewed before final claims.

### Source directories

| Source | Path |
| --- | --- |
| realwaste | `/home/hjc/coSpace/DLA_Final/vision/data/training/realwaste_yolo_cls_train_balanced` |
| tidy | `/home/hjc/coSpace/DLA_Final/vision/data/training/tidy_general_trash_yolo_cls` |
| hf | `/home/hjc/coSpace/DLA_Final/vision/data/training/hf_general_trash_sample_yolo_cls` |

### Split/class counts

| Split | Accept | Reject | Total |
| --- | ---: | ---: | ---: |
| train | 970 | 1180 | 2150 |
| val | 209 | 1689 | 1898 |
| test | 206 | 1688 | 1894 |
| total | 1385 | 4557 | 5942 |

### Source counts

| Source | Split | Accept | Reject | Total |
| --- | --- | ---: | ---: | ---: |
| hf | train | 271 | 542 | 813 |
| hf | val | 59 | 1083 | 1142 |
| hf | test | 57 | 1082 | 1139 |
| realwaste | train | 569 | 569 | 1138 |
| realwaste | val | 122 | 591 | 713 |
| realwaste | test | 122 | 591 | 713 |
| tidy | train | 130 | 69 | 199 |
| tidy | val | 28 | 15 | 43 |
| tidy | test | 27 | 15 | 42 |

## Training setup

- Base model: `runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- Full base model path recorded in args: `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- Data path recorded in args: `/home/hjc/coSpace/DLA_Final/vision/data/training/hf_no_taco_combined_yolo_cls`
- Save dir recorded in args: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001`

### Observed command

```bash
uv run --with ultralytics --with torch --with torchvision --with onnx --with onnxruntime --with onnxslim python scripts/train-yolo-cls.py --data data/training/hf_no_taco_combined_yolo_cls --model runs/general-trash-positive/general-trash-positive-002/weights/best.pt --project runs/hf-no-taco-finetune --serial-prefix hf-no-taco --epochs 25 --batch 32 --device 0 --workers 4 --patience 8 --export-onnx
```

### Key args

| Field | Value |
| --- | --- |
| task/mode | classify / train |
| epochs | 25 |
| patience | 8 |
| batch | 32 |
| image size | 224 |
| device | `0` |
| workers | 4 |
| seed | 20260602 |
| deterministic | true |
| pretrained | true |
| optimizer | auto |
| amp | true |
| export request | `--export-onnx` observed in process command |

## Artifact status

| Artifact | Status | Observed details |
| --- | --- | --- |
| `runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.pt` | present | 3,186,690 bytes; mtime 2026-06-04 12:26:43 CST |
| `runs/hf-no-taco-finetune/hf-no-taco-001/weights/last.pt` | present | 3,186,690 bytes; mtime 2026-06-04 12:26:43 CST |
| `runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.onnx` | present | 6,161,072 bytes; mtime 2026-06-04 12:26:48 CST |
| `runs/hf-no-taco-finetune/hf-no-taco-001/results.csv` | present/complete | last observed epoch: 25 |
| `runs/hf-no-taco-finetune/hf-no-taco-001/training_summary.json` | present | 836 bytes; mtime 2026-06-04 12:26:48 CST |

## Training summary

Current status: completed 25 epochs as of 2026-06-04 12:27:23 CST.

Training summary artifact fields:

| Field | Value |
| --- | --- |
| run_name | `hf-no-taco-001` |
| serial_prefix | `hf-no-taco` |
| epochs | 25 |
| imgsz | 224 |
| batch | 32 |
| device | `0` |
| cuda_available | true |
| cuda_device | NVIDIA GeForce RTX 3060 |
| best_pt | `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.pt` |
| last_pt | `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001/weights/last.pt` |
| export_onnx | `/home/hjc/coSpace/DLA_Final/vision/runs/hf-no-taco-finetune/hf-no-taco-001/weights/best.onnx` |

Observed `results.csv` rows:

| Epoch | Time | Train loss | Val top1 acc | Val top5 acc | Val loss | LR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.3368 | 0.40361 | 0.90253 | 1.00000 | 0.27483 | 0.000547495 |
| 2 | 24.3163 | 0.38335 | 0.85669 | 1.00000 | 0.31213 | 0.001059480 |
| 3 | 35.0623 | 0.38275 | 0.89779 | 1.00000 | 0.30820 | 0.001527450 |
| 4 | 45.3440 | 0.44476 | 0.89779 | 1.00000 | 0.26720 | 0.001468960 |
| 5 | 55.4329 | 0.41509 | 0.90411 | 1.00000 | 0.25035 | 0.001402950 |
| 6 | 65.5225 | 0.37437 | 0.91149 | 1.00000 | 0.22425 | 0.001336930 |
| 7 | 76.4546 | 0.36650 | 0.90358 | 1.00000 | 0.25422 | 0.001270920 |
| 8 | 86.7637 | 0.32929 | 0.89726 | 1.00000 | 0.26624 | 0.001204910 |
| 9 | 96.9432 | 0.31839 | 0.86091 | 1.00000 | 0.31897 | 0.001138890 |
| 10 | 106.9320 | 0.30932 | 0.91623 | 1.00000 | 0.20974 | 0.001072880 |
| 11 | 117.0220 | 0.31165 | 0.85353 | 1.00000 | 0.38427 | 0.001006870 |
| 12 | 127.1520 | 0.28648 | 0.86723 | 1.00000 | 0.33404 | 0.000940855 |
| 13 | 138.3290 | 0.30147 | 0.87039 | 1.00000 | 0.33095 | 0.000874842 |
| 14 | 148.1200 | 0.26168 | 0.86828 | 1.00000 | 0.37021 | 0.000808828 |
| 15 | 158.6340 | 0.26120 | 0.90727 | 1.00000 | 0.22890 | 0.000742815 |
| 16 | 170.4720 | 0.24718 | 0.92255 | 1.00000 | 0.19776 | 0.000676802 |
| 17 | 181.5270 | 0.22350 | 0.88672 | 1.00000 | 0.33603 | 0.000610789 |
| 18 | 192.6760 | 0.21882 | 0.87460 | 1.00000 | 0.33761 | 0.000544776 |
| 19 | 202.9610 | 0.23325 | 0.92571 | 1.00000 | 0.20065 | 0.000478762 |
| 20 | 213.3360 | 0.20641 | 0.84773 | 1.00000 | 0.39515 | 0.000412749 |
| 21 | 223.0210 | 0.22424 | 0.88778 | 1.00000 | 0.30056 | 0.000346736 |
| 22 | 234.5230 | 0.17674 | 0.90674 | 1.00000 | 0.24611 | 0.000280723 |
| 23 | 244.3060 | 0.18460 | 0.90622 | 1.00000 | 0.25798 | 0.000214710 |
| 24 | 254.8410 | 0.15595 | 0.90253 | 1.00000 | 0.27439 | 0.000148696 |
| 25 | 265.0420 | 0.15986 | 0.91675 | 1.00000 | 0.23671 | 0.000082683 |

Best observed val top1 accuracy: 0.92571 at epoch 19.
Best observed val loss: 0.19776 at epoch 16.
Final epoch metrics: train loss 0.15986, val top1 accuracy 0.91675, val top5 accuracy 1.00000, val loss 0.23671.

## Evaluation results

Initial expectation from the parent task was 7 evaluation groups. Actual observed run-local artifacts contain 8 groups, all completed:

| Evaluation group | Count | Weak agreement | Accept n | Reject n | Default threshold | Gate accept recall | Reject false accept rate | Blocked reject rate | Files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hf_sample_test | 1139 | 0.9298 | 57 | 1082 | 0.76 | 0.4912 | 0.0250 | 0.9750 | `hf_sample_test_summary.json`, `hf_sample_test_predictions.csv`, `hf_sample_test_contact_sheet.jpg` |
| hf_combined_test | 2074 | 0.8891 | 324 | 1750 | 0.76 | 0.5679 | 0.0429 | 0.9571 | `hf_combined_test_summary.json`, `hf_combined_test_predictions.csv`, `hf_combined_test_contact_sheet.jpg` |
| hf_no_taco_combined_test | 1894 | 0.9240 | 206 | 1688 | 0.76 | 0.7087 | 0.0320 | 0.9680 | `hf_no_taco_combined_test_summary.json`, `hf_no_taco_combined_test_predictions.csv`, `hf_no_taco_combined_test_contact_sheet.jpg` |
| taco_full | 1451 | 0.5072 | 783 | 668 | 0.76 | 0.3257 | 0.3323 | 0.6677 | `taco_full_summary.json`, `taco_full_predictions.csv`, `taco_full_contact_sheet.jpg` |
| taco_reject_safety | 90 | 0.6778 | 10 | 80 | 0.76 | 0.4000 | 0.2125 | 0.7875 | `taco_reject_safety_summary.json`, `taco_reject_safety_predictions.csv`, `taco_reject_safety_contact_sheet.jpg` |
| tidy_test | 42 | 0.7143 | 27 | 15 | 0.76 | 0.6667 | 0.2667 | 0.7333 | `tidy_test_summary.json`, `tidy_test_predictions.csv`, `tidy_test_contact_sheet.jpg` |
| realwaste_full_test | 713 | 0.9271 | 122 | 591 | 0.76 | 0.8197 | 0.0389 | 0.9611 | `realwaste_full_test_summary.json`, `realwaste_full_test_predictions.csv`, `realwaste_full_test_contact_sheet.jpg` |
| taiwan_mapped | 50 | 0.5200 | 25 | 25 | 0.76 | 0.0000 | 0.0000 | 1.0000 | `taiwan_mapped_summary.json`, `taiwan_mapped_predictions.csv`, `taiwan_mapped_contact_sheet.jpg` |

Top mistakes by summary artifact:

| Evaluation group | Top mistakes |
| --- | --- |
| hf_sample_test | `reject:->accept: 61`; `accept:->reject: 19` |
| hf_combined_test | `reject:->accept: 127`; `accept:->reject: 103` |
| hf_no_taco_combined_test | `reject:->accept: 100`; `accept:->reject: 44` |
| taco_full | `accept:Plastic film->reject: 115`; `accept:Cigarette->reject: 76`; `reject:Clear plastic bottle->accept: 65` |
| taco_reject_safety | `reject:Drink can->accept: 4`; `reject:Clear plastic bottle->accept: 3`; `reject:Glass bottle->accept: 3` |
| tidy_test | `accept:plastic->reject: 5`; `reject:paper->accept: 3`; `accept:plastic-bag->reject: 2` |
| realwaste_full_test | `reject:Plastic->accept: 14`; `accept:Miscellaneous Trash->reject: 12`; `reject:Paper->accept: 9` |
| taiwan_mapped | `accept:general_trash_scene->reject: 23`; `reject:recycling_scene->accept: 1` |

## Threshold sweep

Status: complete for the 8 observed evaluation groups. Thresholds tested: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.76, 0.80, 0.85, 0.90, 0.95.

| Evaluation group | Thresholds | Default 0.76 gate acc | Default recall | Default reject false accept | Best gate acc threshold | Best gate acc | Lowest reject false accept threshold | Lowest reject false accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_sample_test | 11 | 0.9508 | 0.4912 | 0.0250 | 0.90 | 0.9596 | 0.95 | 0.0074 |
| hf_combined_test | 11 | 0.8963 | 0.5679 | 0.0429 | 0.85 | 0.8987 | 0.95 | 0.0166 |
| hf_no_taco_combined_test | 11 | 0.9398 | 0.7087 | 0.0320 | 0.90 | 0.9467 | 0.95 | 0.0113 |
| taco_full | 11 | 0.4831 | 0.3257 | 0.3323 | 0.50 | 0.5072 | 0.95 | 0.1183 |
| taco_reject_safety | 11 | 0.7444 | 0.4000 | 0.2125 | 0.95 | 0.8333 | 0.95 | 0.0750 |
| tidy_test | 11 | 0.6905 | 0.6667 | 0.2667 | 0.70 | 0.7143 | 0.95 | 0.1333 |
| realwaste_full_test | 11 | 0.9369 | 0.8197 | 0.0389 | 0.95 | 0.9425 | 0.95 | 0.0152 |
| taiwan_mapped | 11 | 0.5000 | 0.0000 | 0.0000 | 0.55 | 0.5400 | 0.55 | 0.0000 |

## Comparison with previous HF+TACO run

Baseline run used for comparison: `runs/general-trash-positive/general-trash-positive-002`.

Baseline metadata from `training_summary.json`:

| Field | Value |
| --- | --- |
| run_name | `general-trash-positive-002` |
| serial_prefix | `general-trash-positive` |
| data | `/home/hjc/coSpace/DLA_Final/vision/data/training/general_trash_positive_relaxed_yolo_cls` |
| base_model | `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-001/weights/best.pt` |
| best_pt | `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/best.pt` |
| last_pt | `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/last.pt` |
| export_onnx | `/home/hjc/coSpace/DLA_Final/vision/runs/general-trash-positive/general-trash-positive-002/weights/best.onnx` |
| epochs/imgsz/batch/device | 25 / 224 / 32 / `0` |

Comparison on the 5 common evaluation groups, using the default threshold metrics in each summary:

| Group | HF+TACO count | no-TACO count | Agreement old | Agreement new | Delta | Recall old | Recall new | Delta | Reject FAR old | Reject FAR new | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| taco_full | 1451 | 1451 | 0.5996 | 0.5072 | -0.0924 | 0.2656 | 0.3257 | +0.0600 | 0.1168 | 0.3323 | +0.2156 |
| taco_reject_safety | 90 | 90 | 0.5222 | 0.6778 | +0.1556 | 0.2000 | 0.4000 | +0.2000 | 0.0750 | 0.2125 | +0.1375 |
| tidy_test | 42 | 42 | 0.5714 | 0.7143 | +0.1429 | 0.4815 | 0.6667 | +0.1852 | 0.1333 | 0.2667 | +0.1333 |
| realwaste_full_test | 713 | 713 | 0.9130 | 0.9271 | +0.0140 | 0.8197 | 0.8197 | +0.0000 | 0.0440 | 0.0389 | -0.0051 |
| taiwan_mapped | 50 | 50 | 0.5600 | 0.5200 | -0.0400 | 0.2400 | 0.0000 | -0.2400 | 0.0400 | 0.0000 | -0.0400 |

Interpretation of common-group comparison:

- no-TACO improved weak-label agreement on `taco_reject_safety`, `tidy_test`, and slightly on `realwaste_full_test`.
- no-TACO worsened weak-label agreement on `taco_full` and `taiwan_mapped`.
- no-TACO increased accept recall on `taco_full`, `taco_reject_safety`, and `tidy_test`, but that came with materially higher reject false-accept rates on the same three sets.
- `realwaste_full_test` is the cleanest common-set improvement: agreement increased slightly and reject false-accept rate decreased slightly, with unchanged accept recall.
- `taiwan_mapped` became much more conservative at threshold 0.76: reject false-accept rate fell to 0.0000, but accept recall also fell to 0.0000.

Non-common evaluation artifacts:

| Run | Evaluation group | Count | Agreement | Recall | Reject FAR |
| --- | --- | ---: | ---: | ---: | ---: |
| HF+TACO previous only | relaxed_test | 935 | 0.8353 | 0.5169 | 0.0554 |
| HF+TACO previous only | combined_test | 973 | 0.8109 | 0.5169 | 0.0609 |
| no-TACO current only | hf_sample_test | 1139 | 0.9298 | 0.4912 | 0.0250 |
| no-TACO current only | hf_combined_test | 2074 | 0.8891 | 0.5679 | 0.0429 |
| no-TACO current only | hf_no_taco_combined_test | 1894 | 0.9240 | 0.7087 | 0.0320 |

No threshold sweep artifacts were observed in `runs/general-trash-positive/general-trash-positive-002`, so sweep-to-sweep comparison is not available from current artifacts.

## Issues and caveats

- This record is based on completed run-local artifacts observed through threshold sweep completion.
- Dataset summary labels the merged dataset as weak-label and explicitly notes that labels should be reviewed before final claims.
- The dataset is class-imbalanced, especially in validation and test splits (`reject` dominates).
- The parent task mentioned 7 evaluation groups, but the observed evaluation command and artifacts produced 8 groups: `hf_sample_test`, `hf_combined_test`, `hf_no_taco_combined_test`, `taco_full`, `taco_reject_safety`, `tidy_test`, `realwaste_full_test`, and `taiwan_mapped`.
- Evaluation labels are weak or mapped labels according to the summary note; results need human review before final claims.
- `taco_full`, `tidy_test`, and `taiwan_mapped` show weak agreement near or below 0.72 at the default threshold, so these are not strong positive validation signals without label audit.
- `taiwan_mapped` has zero accept recall at threshold 0.76, while also showing zero reject false accept rate; this is a conservative but likely over-blocking behavior on that mapped set.
- The HF+TACO comparison is limited to shared evaluation group names and default-threshold summary metrics. The current no-TACO run has HF-specific evaluation groups that are not directly comparable to the previous `combined_test` and `relaxed_test` artifacts.
- Previous HF+TACO threshold sweeps were not present in the observed baseline run directory, so only current no-TACO threshold behavior is recorded.
- ONNX export completed and was verified by the presence of `weights/best.onnx`.
