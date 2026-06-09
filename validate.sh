#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" scripts/validate-vision-contract.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_model_adapter.py
PYTHONPATH=src "$PYTHON_BIN" tests/smoke_stub_inference.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_accept_gate.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_voice_feedback.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_agx_audio.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_agx_l515_voice_demo.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_esp32_serial.py
PYTHONPATH=src "$PYTHON_BIN" tests/test_runtime_integration.py
"$PYTHON_BIN" scripts/validate-model-metrics.py
"$PYTHON_BIN" tests/test_weak_dataset_mapping.py
"$PYTHON_BIN" tests/test_build_yolo_cls_dataset.py
"$PYTHON_BIN" tests/test_build_folder_yolo_cls_dataset.py
"$PYTHON_BIN" tests/test_merge_yolo_cls_datasets.py
"$PYTHON_BIN" tests/test_train_serial_names.py
"$PYTHON_BIN" tests/test_threshold_sweep.py
"$PYTHON_BIN" tests/test_l515_demo_experiment.py
"$PYTHON_BIN" tests/test_download_hf_class_sample.py
"$PYTHON_BIN" tests/test_prepare_user_accept_seed.py
"$PYTHON_BIN" tests/test_demo_accept_candidate_eval.py
