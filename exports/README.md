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

## Demo Accept Recall Run

`20260608-demo-accept-recall` is the user accept seed fine-tune export for the
June 2026 accept-only demo. It should not replace the recommended production
baseline.

Use it only with the demo gate:

- `accept_threshold`: `0.50`
- `uncertain_threshold`: `0.50`

Observed weak-label metrics:

- user seed holdout at threshold `0.76`: `8/9` accepted, gate accept recall `0.8888888888888888`
- HF combined test at threshold `0.50`: gate accept recall `0.7716049382716049`, reject false accept rate `0.09314285714285714`
- TACO full at threshold `0.76`: gate accept recall `0.39719029374201786`, reject false accept rate `0.218562874251497`

Prepared demo props should still pass `scripts/run-demo-accept-candidate-eval.py`
with at least `2/3` shots accepted before they are used on stage.

## Checksums

```text
aaa29a5a510b850fc1d1a9cc30574cf48e0ea6239561d7ec2111037d8112cf32  exports/20260601T122805Z/best.pt
f80c72076b4850c8a7a5742814c0a2fd79a3daf2ec8bba143133d16c33daa19e  exports/20260601T122805Z/best.onnx
53bcf4e1933a903c86a987ab8a95ad0843f61002d7d0ac23326c53306ed4c0d6  exports/20260601T144442Z-hard-negative/best.pt
7581f54aeee23c118b8a40b9fc6fef77d03477c2ec91bfe51262281d21eb8fa0  exports/20260601T144442Z-hard-negative/best.onnx
eaad1a0a97a1fdf18a295bf7cfb21effebde5342eeb1d610f74277da30e99d9a  exports/20260608-demo-accept-recall/best.pt
1f9b3aeec2540e61f64777f418f103e8e8ae00be8a3ae242881a8f38d7d4384c  exports/20260608-demo-accept-recall/best.onnx
```
