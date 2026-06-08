# User Accept Seed Fine-tune Report

## Status

This is a normal-camera accept seed experiment only. It does not update AGX/runtime defaults, does not replace anything under `exports/`, and does not change the v0.3 center contract or `recognition_result` payload.

Decision: keep as experiment results, not a deployment candidate. The seed holdout improved sharply, but the holdout is only 9 images and broad external accept recall regressed on several datasets.

## Data

- Source zip: `Dataset-20260608T092356Z-3-001.zip`
- Source image count: 55 normal-camera screenshots
- Label policy: all 55 images are weak/user-confirmed `accept` general trash
- Source output: `data/sources/user_accept_seed_20260608/`
- Source contact sheet: `data/sources/user_accept_seed_20260608/contact_sheet.jpg`
- Seed YOLO cls dataset: `data/training/user_accept_seed_yolo_cls`
- Seed counts after repeat: train accept 304, val accept 8, test accept 9; reject folders intentionally empty
- Source counts before repeat: train accept 38, val accept 8, test accept 9
- Combined dataset: `data/training/user_accept_seed_combined_yolo_cls`, total 7463 images
- Combined train counts: accept 1822, reject 1472

## Training

- Base model: `/home/hjc/coSpace/DLA_Final/vision/runs/hf-general-trash-finetune/hf-general-trash-001/weights/best.pt`
- Run directory: `/home/hjc/coSpace/DLA_Final/vision/runs/user-accept-seed-finetune/user-accept-seed-001`
- Best PT: `/home/hjc/coSpace/DLA_Final/vision/runs/user-accept-seed-finetune/user-accept-seed-001/weights/best.pt`
- ONNX: `/home/hjc/coSpace/DLA_Final/vision/runs/user-accept-seed-finetune/user-accept-seed-001/weights/best.onnx`
- Requested config: epochs 15, batch 32, device 0, workers 4, patience 5, imgsz 224
- Actual result: early-stopped after 11 epochs; best epoch 6 with val top1 87.8% and val loss 0.2998
- CUDA device: NVIDIA GeForce RTX 3060

## Gate Metrics At accept_threshold=0.76

| Eval set | New model | Base comparison |
| --- | --- | --- |
| User accept seed test | count 9; accept 8/9 (88.9%); reject false accept 0/0 (n/a) | base user seed accept recall 22.2%; delta +66.7pp |
| HF sample test | count 1139; accept 34/57 (59.6%); reject false accept 34/1082 (3.1%) | accept recall +12.3pp; reject false accept +0.3pp |
| HF combined test | count 2074; accept 174/324 (53.7%); reject false accept 66/1750 (3.8%) | accept recall -5.2pp; reject false accept -0.6pp |
| TACO full accept-focus | count 1451; accept 311/783 (39.7%); reject false accept 146/668 (21.9%) | accept recall -14.3pp; reject false accept -6.9pp |
| TACO reject-safety | count 90; accept 4/10 (40.0%); reject false accept 8/80 (10.0%) | accept recall +10.0pp; reject false accept -7.5pp |
| TIDY test | count 42; accept 17/27 (63.0%); reject false accept 2/15 (13.3%) | accept recall +0.0pp; reject false accept -6.7pp |
| RealWaste full test | count 713; accept 79/122 (64.8%); reject false accept 15/591 (2.5%) | accept recall -13.1pp; reject false accept -1.5pp |
| Taiwan mapped weak set | count 50; accept 1/25 (4.0%); reject false accept 1/25 (4.0%) | accept recall +0.0pp; reject false accept +0.0pp |

## Assessment

- User seed holdout improved from 2/9 accepted to 8/9 accepted, so the fine-tune did learn the provided normal-camera examples.
- It still misses the strict 90% seed holdout target by one image: 8/9 is 88.9%. With only 9 holdout images, this result is too brittle for deployment judgment.
- Safety did not broadly regress: TACO reject-safety false accept improved from 17.5% to 10.0%, TIDY reject false accept improved from 20.0% to 13.3%, and RealWaste reject false accept improved from 4.1% to 2.5%.
- Broad accept recall regressed on important external sets: TACO full accept recall fell from 54.0% to 39.7%, RealWaste full accept recall fell from 77.9% to 64.8%, and HF combined accept recall fell from 59.0% to 53.7%.
- Recommendation: keep `user-accept-seed-001` as an experiment artifact. Do not deploy or mark as reviewed. Next useful step is to collect more accept-positive normal-camera/L515 crops and rebalance fine-tune so the new examples improve soft-trash recall without collapsing broader accept coverage.

## Artifacts

- Per-set predictions, summaries, contact sheets, and threshold sweeps: `runs/user-accept-seed-finetune/user-accept-seed-001/weak_eval/`
- Report path: `runs/user-accept-seed-finetune/user_accept_seed_report.md`

## Validation

- `python3 -m py_compile scripts/*.py tests/*.py`: passed
- `./validate.sh`: passed
