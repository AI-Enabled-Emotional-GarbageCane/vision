# Training Lineage

This document records the model experiments and artifacts that are kept in this
repository. Source datasets remain excluded from git under `data/`, but the run
metadata, reports, predictions, contact sheets, and small model weights are
tracked for handoff and review.

## Policy

- Tracked: `exports/`, `runs/`, reports, predictions, contact sheets, `.pt`, `.onnx`, and small local model artifacts.
- Not tracked: raw datasets under `data/`, ad hoc demo candidate photos under `demo_candidates/`, and raw dataset archives such as `Dataset-*.zip`.
- Runtime contract remains binary `accept` / `reject`; demo-only thresholds do not change production defaults.

## Curated Exports

| Export | Role | Dataset / Source | Training / Selection | Artifact |
|---|---|---|---|---|
| `exports/20260601T122805Z` | public-dataset PoC baseline | TrashNet + RealWaste binary remap | 50 epochs, batch 32, imgsz 224; public test top1 `0.856269121170044` | `best.pt`, `best.onnx`, `metrics.json` |
| `exports/20260601T144442Z-hard-negative` | hard-negative experiment | hard-negative public dataset experiment | Colab interrupted after epoch 9; selected gate threshold `0.68`; not recommended | `best.pt`, `best.onnx`, `metrics.json` |
| `exports/20260608-demo-accept-recall` | accept-only demo model | `user-accept-seed-001` export | demo gate `accept_threshold=0.50`; not production-ready | `best.pt`, `best.onnx`, `metrics.json`, `demo_config.json` |

See `exports/README.md` and `docs/model-registry.md` for checksums and curated export notes.

## Local Training Runs

| Run | Purpose | Base Model | Dataset | Key Parameters | Main Outputs |
|---|---|---|---|---|---|
| `runs/realwaste-accuracy/realwaste-accuracy-001` | RealWaste accuracy experiment | `exports/20260601T122805Z/best.pt` | `data/training/realwaste_yolo_cls` | 30 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `training_summary.json` |
| `runs/realwaste-accuracy/realwaste-accuracy-002` | RealWaste full weak-label experiment | `exports/20260601T122805Z/best.pt` | `data/training/realwaste_yolo_cls_full` | 30 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, weak eval outputs |
| `runs/reject-safety/local-rtx3060-reject-safety-20260602` | reject-safety hard-negative experiment | `exports/20260601T122805Z/best.pt` | `data/training/reject_safety_yolo_cls` | 25 epochs, batch 16, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, weak eval outputs |
| `runs/general-trash-positive/general-trash-positive-001` | first general-trash positive fine-tune | `runs/realwaste-accuracy/realwaste-accuracy-002/weights/best.pt` | `data/training/general_trash_positive_combined_yolo_cls` | 30 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `fine_tune_report.md` |
| `runs/general-trash-positive/general-trash-positive-002` | relaxed general-trash positive fine-tune | `runs/general-trash-positive/general-trash-positive-001/weights/best.pt` | `data/training/general_trash_positive_relaxed_yolo_cls` | 25 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `fine_tune_report.md` |
| `runs/hf-general-trash-finetune/hf-general-trash-001` | HF + public combined fine-tune | `runs/general-trash-positive/general-trash-positive-002/weights/best.pt` | `data/training/hf_general_trash_combined_yolo_cls` | 25 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `hf_general_trash_finetune_report.md` |
| `runs/hf-no-taco-finetune/hf-no-taco-001` | no-TACO comparison | `runs/general-trash-positive/general-trash-positive-002/weights/best.pt` | `data/training/hf_no_taco_combined_yolo_cls` | 25 epochs, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `hf_no_taco_finetune_report.md` |
| `runs/user-accept-seed-finetune/user-accept-seed-001` | normal-camera accept seed demo fine-tune | `runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.pt` | `data/training/user_accept_seed_combined_yolo_cls` | requested 15 epochs, early-stopped after 11, best epoch 6, batch 32, imgsz 224, device 0 | `weights/best.pt`, `weights/best.onnx`, `user_accept_seed_report.md` |

Each run directory keeps its own `training_summary.json` where available. These
summaries are the primary source for exact paths and training parameters.

## Evaluation-Only / Demo Runs

| Path | Purpose | Notes |
|---|---|---|
| `runs/hf-general-trash-eval` | HF dataset trial and model comparisons | Contains reports and weak-label evaluation outputs. |
| `runs/l515-demo-positive/l515_demo_fix_report.md` | L515 demo plan/report record | Created for the L515 accept-positive repair plan; fine-tune should only run when enough L515 raw samples exist. |
| `runs/demo-accept-recall` | accept-only demo candidate screening | Contains demo config/output directories; candidate photos remain under ignored `demo_candidates/`. |

## Dataset Lineage

- `realwaste_yolo_cls*`: RealWaste-derived weak binary mapping; source data remains under ignored `data/sources/realwaste`.
- `general_trash_positive*`: TACO accept-focus plus TIDY positive/negative remaps for soft wrappers, tissues, cigarette butts, plastic bags, and small trash.
- `hf_general_trash*`: Hugging Face waste datasets merged with public hard negatives and positive trash mappings.
- `hf_no_taco_combined_yolo_cls`: HF + RealWaste + TIDY comparison without TACO.
- `user_accept_seed*`: 55 user-provided normal-camera screenshots, all mapped to weak/user-confirmed `accept`; source zip is ignored.

Review the corresponding run report before using any non-recommended model for
deployment. Several experiments intentionally increase accept recall and are not
selected for reject safety.
