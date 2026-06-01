# Model Exports

This directory contains curated small Vision v0.3 model exports that are intentionally tracked for project handoff.

Raw datasets, full training runs, TensorRT engines, and ad hoc generated artifacts still stay out of git.

## Recommended Model

Use `20260601T122805Z` as the current public-dataset PoC baseline:

- `best.pt`: YOLOv11n classification PyTorch checkpoint
- `best.onnx`: ONNX export
- `metrics.json`: public test metrics
- `sample_recognition_result.json`: contract-shaped inference sample

This run reached public test top-1 accuracy `0.856269121170044`. With the deployment accept gate at threshold `0.76`, its false accept rate on reject samples was `0.09815950920245399`.

## Experimental Run

`20260601T144442Z-hard-negative` is a partial hard-negative experiment. Colab disconnected after epoch 9, so this run should not replace the recommended baseline.

Observed test metrics:

- raw top-1 accuracy: `0.8746177370030581`
- raw false accept rate on reject samples: `0.20245398773006135`
- gate accuracy at selected threshold `0.68`: `0.8685015290519877`
- gate false accept rate on reject samples: `0.1656441717791411`

It improved raw top-1 accuracy but did not improve the false accept risk.

## Checksums

```text
aaa29a5a510b850fc1d1a9cc30574cf48e0ea6239561d7ec2111037d8112cf32  exports/20260601T122805Z/best.pt
f80c72076b4850c8a7a5742814c0a2fd79a3daf2ec8bba143133d16c33daa19e  exports/20260601T122805Z/best.onnx
53bcf4e1933a903c86a987ab8a95ad0843f61002d7d0ac23326c53306ed4c0d6  exports/20260601T144442Z-hard-negative/best.pt
7581f54aeee23c118b8a40b9fc6fef77d03477c2ec91bfe51262281d21eb8fa0  exports/20260601T144442Z-hard-negative/best.onnx
```
