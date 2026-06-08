# Demo Accept Recall Export

This export is for the June 2026 accept-only demo flow. It is intentionally
tracked so the demo machine can fetch the same fine-tuned model from git.

## Files

- `best.pt`: YOLOv11n classification PyTorch checkpoint from
  `runs/user-accept-seed-finetune/user-accept-seed-001`
- `best.onnx`: ONNX export of the same checkpoint
- `metrics.json`: demo gate, provenance, and weak-label evaluation notes
- `demo_config.json`: demo-only runtime settings

## Use

Use this only for accept-only demo screening:

```bash
uv run --with onnxruntime --with pillow --with numpy \
  python scripts/run-demo-accept-candidate-eval.py \
  --model exports/20260608-demo-accept-recall/best.onnx \
  --accept-threshold 0.50 \
  --uncertain-threshold 0.50 \
  --enforce-smoke
```

This model should not replace the production recommended export. It prioritizes
accept recall for prepared general-trash props and was not selected for reject
safety.
