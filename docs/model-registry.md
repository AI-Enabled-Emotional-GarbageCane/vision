# Vision Model Registry

This file records the curated model exports currently present in `vision/exports/`.

## Label Policy

The public payload remains binary:

- `accept`: general waste / combustible waste accepted by this bin.
- `reject`: objects that should not enter this bin, such as recyclable paper, rigid plastic containers, metal, glass, and food waste.

Important demo clarification: flexible candy wrappers, snack bags, dirty plastic film, and mixed-material flexible packaging should be labeled `accept` if the product policy treats them as general waste. Do not collapse all plastic into `reject`.

## AGX L515 Smoke Sample

The same L515 RGB snapshot was used to compare all ONNX exports:

```text
/home/dla_test/DLA_Final/vision/snapshots/l515-20260604T011500.jpg
```

Object: candy wrapper / flexible plastic packaging.

Expected label for the current demo discussion: `accept`.

Only ONNX artifacts were executed on AGX in this smoke test. PyTorch `.pt` artifacts are recorded by path and sha256, but were not executed because this AGX Python environment does not currently have `torch` / `ultralytics` installed.

## Export: `20260601T122805Z`

| Field | Value |
|---|---|
| Role | Default public baseline |
| Default adapter selection | Yes |
| Model type | YOLOv11n classification |
| Classes | `accept`, `reject` |
| Input size | `224` |
| Epochs | `50` |
| Dataset | TrashNet + RealWaste binary remap |
| Test top-1 | `0.856269121170044` |
| Top-1 target | `0.85` |
| Target met | `true` |
| ONNX artifact | `exports/20260601T122805Z/best.onnx` |
| ONNX sha256 | `f80c72076b4850c8a7a5742814c0a2fd79a3daf2ec8bba143133d16c33daa19e` |
| PyTorch artifact | `exports/20260601T122805Z/best.pt` |
| PyTorch sha256 | `aaa29a5a510b850fc1d1a9cc30574cf48e0ea6239561d7ec2111037d8112cf32` |

Dataset counts:

| Split | accept | reject |
|---|---:|---:|
| train | 760 | 761 |
| val | 163 | 163 |
| test | 164 | 163 |

AGX L515 candy-wrapper result:

| Prediction | Confidence | Deployment `allow_accept` | Expected |
|---|---:|---:|---:|
| `reject` | `0.988198459148407` | `False` | `accept` |

Assessment:

- This is the current integration baseline and remains useful for exercising `recognition_result`.
- It fails the candy-wrapper general-waste case as a high-confidence false reject.
- Threshold tuning cannot fix this case because the predicted class is `reject`.

## Export: `20260601T144442Z-hard-negative`

| Field | Value |
|---|---|
| Role | Hard-negative experiment |
| Default adapter selection | No |
| Model type | YOLOv11n classification |
| Classes | `accept`, `reject` |
| Baseline run | `20260601T122805Z` |
| Raw top-1 | `0.8746177370030581` |
| Top-1 target | `0.9` |
| Top-1 target met | `false` |
| Raw false accept rate on reject | `0.20245398773006135` |
| Selected accept threshold | `0.68` |
| Gate accuracy | `0.8685015290519877` |
| Gate false accept rate on reject | `0.1656441717791411` |
| Gate reject recall | `0.8343558282208589` |
| False-accept target met | `false` |
| ONNX artifact | `exports/20260601T144442Z-hard-negative/best.onnx` |
| ONNX sha256 | `7581f54aeee23c118b8a40b9fc6fef77d03477c2ec91bfe51262281d21eb8fa0` |
| PyTorch artifact | `exports/20260601T144442Z-hard-negative/best.pt` |
| PyTorch sha256 | `53bcf4e1933a903c86a987ab8a95ad0843f61002d7d0ac23326c53306ed4c0d6` |

AGX L515 candy-wrapper result:

| Prediction | Confidence | Deployment `allow_accept` | Expected |
|---|---:|---:|---:|
| `reject` | `0.9892927408218384` | `False` | `accept` |

Assessment:

- This experiment improves raw top-1 versus the baseline, but it does not meet the false-accept safety target.
- It does not fix the candy-wrapper general-waste case; it also returns a high-confidence false reject.
- Keep it as an experiment record, not as the default deployment export.

## Current Recommendation

Retrain/fine-tune after clarifying label policy for flexible plastic packaging:

- Add L515 demo-angle `accept` samples for candy wrappers, snack bags, plastic film, and mixed-material flexible packaging.
- Add L515 demo-angle `reject` samples for rigid plastic bottles/containers and other recyclable plastic items.
- Keep the current default export as an integration baseline only.
- Do not switch to the hard-negative export for deployment; it does not improve the observed AGX L515 candy-wrapper issue.
