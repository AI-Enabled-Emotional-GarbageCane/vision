# AGX L515 Vision Integration Log - 2026-06-04

## Scope

本紀錄保存 AGX + Intel RealSense L515 + `vision` 模型的實機串接結果，以及糖果包裝塑膠紙測試暴露出的模型/標籤問題。

All curated model export records are maintained in [`model-registry.md`](./model-registry.md).

## Environment

- Device: Jetson AGX / aarch64
- Camera: Intel RealSense L515, serial `f1272157`, firmware `1.5.8.1`
- RealSense SDK: user-local RSUSB build at `/home/dla_test/.local/realsense-l515-rsusb`
- Python binding: `pyrealsense2 2.54.1`
- Model runtime: `onnxruntime==1.23.2` on CPU provider
- Default model export: `exports/20260601T122805Z/best.onnx`

Required runtime environment:

```bash
export LD_LIBRARY_PATH=/home/dla_test/.local/realsense-l515-rsusb/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/dla_test/DLA_Final/vision/src:/home/dla_test/.local/realsense-l515-rsusb/OFF:$PYTHONPATH
```

When running firmware and vision in the same script, also include:

```bash
export PYTHONPATH=/home/dla_test/DLA_Final/firmware:$PYTHONPATH
```

## Implementation Notes

- `L515ColorCamera` now uses `timeout_ms=3000` and `warmup_frames=30`.
- Reason: L515 RGB frames are nearly black for the first several frames after stream start; brightness stabilizes around frame 30 on this AGX setup.
- The queue contract is unchanged:
  - firmware consumes depth and emits `user_detected` into `q_detected`
  - vision consumes `user_detected`, captures L515 RGB, runs the classifier, saves a snapshot, and emits `recognition_result` into `q_result`
- `vision ./validate.sh` remains green after the runtime warmup change.

## Test Result

Object under test: candy wrapper / flexible plastic packaging.

Expected product label from demo discussion:

- `accept`
- Reason: user identified it as general waste (`一般垃圾`) rather than recyclable plastic.

Observed default model result:

```text
firmware_event:
{'event': 'user_detected', 'distance_cm': 67.6, 'ts': '2026-06-04T01:14:56'}

vision_result:
{'event': 'recognition_result',
 'class': 'reject',
 'confidence': 0.9855062961578369,
 'num_objects': 1,
 'snapshot_path': '/home/dla_test/DLA_Final/vision/snapshots/l515-20260604T011500.jpg',
 'ts': '2026-06-04T01:15:00'}
```

Snapshot:

```text
/home/dla_test/DLA_Final/vision/snapshots/l515-20260604T011500.jpg
```

Cross-check with available ONNX exports on the same snapshot:

| Export | Prediction | Confidence | Deployment allow_accept |
|---|---:|---:|---:|
| `20260601T122805Z/best.onnx` | `reject` | `0.988198459148407` | `False` |
| `20260601T144442Z-hard-negative/best.onnx` | `reject` | `0.9892927408218384` | `False` |

## Model Issue

This is a false reject if flexible candy-wrapper plastic should be treated as general waste for this product.

The issue is not fixable by changing the accept threshold. The model predicts `reject` with very high confidence, and the deployment gate only prevents unsafe false accepts; it cannot rescue high-confidence false rejects.

There is also a label taxonomy ambiguity in the current spec: `accept` is defined as general waste, but `reject` includes "plastic" as a broad category. Flexible plastic wrappers and rigid/recyclable plastic containers need to be separated in the labeling rules before retraining.

## Retraining Recommendation

Retraining/fine-tuning is recommended after clarifying label policy:

- Treat flexible wrappers, candy packaging, snack bags, dirty plastic film, and mixed-material packaging as `accept` if they are expected to go into this general-waste bin.
- Keep recyclable/should-not-enter items such as rigid plastic bottles, clean plastic containers, paper, metal, glass, and food waste as `reject`.
- Collect L515 demo-angle images under the real bucket/camera setup:
  - at least 50-100 images for flexible-wrapper `accept`
  - at least 50-100 images for rigid/recyclable-plastic `reject`
  - add lighting, distance, rotation, partial occlusion, and hand-presented variations
- Keep a small L515-only validation set for final acceptance, separate from public TrashNet/RealWaste validation.

Near-term action: do not ship this model as final for the demo if candy-wrapper general waste must be accepted. Use the current model only as a working integration baseline.
