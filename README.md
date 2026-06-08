# vision

AI 影像辨識服務 — [AI 情緒垃圾筒](https://github.com/AI-Enabled-Emotional-GarbageCane/ai-enabled-emotional-garbagecane) 的視覺子系統。

## 職責

- 影像資料集收集與標註
- 模型訓練（垃圾分類）
- 推論 service:讀取攝影機畫面 → 輸出 `recognition_result`

## 介面

- 接收 firmware 的觸發訊號(`user_detected`)後啟動推論
- 將結果(`recognition_result`)發送給 display，payload 為 `{event, class, confidence, num_objects, snapshot_path, ts}`

詳細 API 契約見 [monorepo `docs/api-contract.md`](https://github.com/AI-Enabled-Emotional-GarbageCane/ai-enabled-emotional-garbagecane/blob/main/docs/api-contract.md)。本 repo 以 `contracts/contract.lock.json` 鎖定 v0.3。

本 repo 內部規格盤點見 [`docs/vision-spec.md`](./docs/vision-spec.md)。

## 技術棧

- YOLOv11n binary classification (`accept` / `reject`)
- Intel RealSense L515 RGB capture
- Jetson AGX Orin Nano + TensorRT deployment target
- Python `multiprocessing.Queue` integration

## 驗證

```bash
./validate.sh
```

預設驗證會跑 contract/spec 檢查、stub model smoke test、accept gate 與 baseline metrics 檢查。大型資料、完整 training runs 與 TensorRT engine 不進 git；已挑選的小型模型交付檔放在 [`exports/`](./exports/)。

目前 public baseline 已加入保守 accept gate：payload 的 `class` 仍保留模型 best guess，但部署放行只在 `class == "accept"` 且 `confidence >= 0.76` 時成立。這用來降低 `reject -> accept` 風險；L515 真實場景仍需另行驗收。

## Runtime 串接

`src/runtime.py` 提供 queue 邊界的實際串接入口：消費 firmware 放入 `q_detected` 的 `user_detected`，擷取 L515 RGB frame，透過模型 adapter 推論，保存 snapshot，再把 `recognition_result` 放入 `q_result`。這層只依賴 public queue payload，不依賴 firmware 內部 class 或 function。

模型 adapter 預設使用 `exports/20260601T122805Z/best.onnx`；若部署環境沒有 ONNX Runtime，可改傳入 PyTorch `.pt` 模型或在測試中注入符合 `ImageClassifier` protocol 的 predictor。

## 語音回饋 handoff

`src/voice_feedback.py` 會把 `recognition_result` 轉成可選的 `voice_feedback_cue`，內容包含 roast 情境、GPT-SoVITS 預生成音檔路徑、台詞、音效與 0.5 秒停頓設定。這是給整合測試或下游 display 使用的 adapter；vision 仍不直接播放語音，也不改 `q_result` 的 v0.3 必要欄位。

若 runtime 傳入 `q_voice`，每次送出 `recognition_result` 後會額外送出一筆 voice cue；未傳入時行為維持原本 contract。
當 `recognition_result.class == "reject"` 且信心高於門檻時，voice cue 會從 30 句 `reject/reject-01.wav` 到 `reject/reject-30.wav` 錄音池隨機選一段。

GPT-SoVITS 模型與語音素材產生放在 AGX / Jetson 端；ESP32 只當播放端。`src/esp32_serial.py` 可把 `voice_feedback_cue` 轉成 ESP32 透過 USB Serial / UART 接收的一行 JSON：

```json
{"category":"reject","audio_path":"reject/reject-01.wav","pre_sfx":"ding","pre_delay_ms":500}
```

若 runtime 傳入 `voice_sink`，同一筆 voice cue 會被送到該 sink；可用 `Esp32SerialVoiceSink("/dev/ttyACM0")` 把指令送給 ESP32-S3 voice player。ESP32 sketch 與 SD card 音檔目錄放在 firmware repo 的 `esp32_voice_player/`。

## 模型紀錄

- 目前所有 curated exports 的紀錄見 [`docs/model-registry.md`](./docs/model-registry.md)。
- 各模型的訓練資料、參數、run 輸出與用途見 [`docs/training-lineage.md`](./docs/training-lineage.md)。
- AGX + L515 實機串接與糖果包裝塑膠紙模型問題見 [`docs/agx-l515-vision-integration-20260604.md`](./docs/agx-l515-vision-integration-20260604.md)。

## Demo accept-only 篩選

明天只展示一般垃圾時，可使用 demo-only low gate，不修改 production default，也不替換 `exports/`。
先把每個道具拍 3 張，放到 `demo_candidates/accept_props/<item_name>/`：

```bash
uv run --with onnxruntime --with pillow --with numpy \
  python scripts/run-demo-accept-candidate-eval.py \
  --model runs/user-accept-seed-finetune/user-accept-seed-001/weights/best.onnx \
  --accept-threshold 0.50 \
  --uncertain-threshold 0.50 \
  --enforce-smoke
```

每個道具至少 2/3 張被 gate 成 `accept` 才會列入
`runs/demo-accept-recall/<run>/demo_accepted_props.txt`。這只代表 accept-only
demo 道具篩選通過，不代表 reject safety 或 production readiness。

## 弱標籤推論資料

目前可用 `scripts/prepare-weak-finetune-dataset.py` 從 TACO 抽樣建立本機弱標籤資料。預設目標是 reject safety：多抽瓶、罐、玻璃、紙盒、塑膠瓶等 `reject` hard negatives，少量 `accept` 只作 sanity check。專題的主要風險指標應優先看 `false_accept_rate_on_reject`，避免把「不是一般垃圾」誤收。

```bash
python3 scripts/prepare-weak-finetune-dataset.py \
  --output-dir data/inference_extra_waste/taco_reject_safety

uv run --with onnxruntime --with pillow --with numpy \
  python scripts/evaluate-weak-manifest.py \
  --dataset-dir data/inference_extra_waste/taco_reject_safety \
  --manifest data/inference_extra_waste/taco_reject_safety/manifest.csv \
  --model exports/20260601T122805Z/best.onnx \
  --output data/inference_extra_waste/taco_reject_safety/predictions.csv \
  --summary data/inference_extra_waste/taco_reject_safety/prediction_summary.json \
  --contact-sheet data/inference_extra_waste/taco_reject_safety/contact_sheet.jpg
```

這些資料的 `eval_label` 是弱標籤，必須人工審閱後才可作為 fine-tune 訓練資料或正式驗收資料。

## 本機 Fine-tune

RTX 3060 12GB 可直接跑本機 YOLOv11n-cls fine-tune。先把弱標籤 manifest 轉成 YOLO classification 目錄，再從目前 baseline `best.pt` 繼續訓練：

```bash
python3 scripts/build-yolo-cls-dataset.py \
  --manifest data/inference_extra_waste/taco_reject_safety/manifest.csv \
  --source-root data/inference_extra_waste/taco_reject_safety \
  --output-dir data/training/reject_safety_yolo_cls

uv run --with ultralytics --with torch --with torchvision \
  --with onnx --with onnxruntime --with onnxslim \
  python scripts/train-yolo-cls.py \
  --data data/training/reject_safety_yolo_cls \
  --model exports/20260601T122805Z/best.pt \
  --project runs/reject-safety \
  --serial-prefix reject-safety \
  --epochs 25 \
  --batch 16 \
  --device 0 \
  --export-onnx
```

`--name` 未指定時會自動使用 `--serial-prefix` 產生下一個流水號目錄，例如
`reject-safety-001`、`reject-safety-002`。訓練權重會留在各自 run 的 `weights/`
底下，不會覆蓋上一輪。

訓練後仍需用 `scripts/evaluate-weak-manifest.py` 重算 `false_accept_rate_on_reject`，確認 reject safety 沒有退步。

## RealWaste 大型弱標籤訓練

RealWaste 可作為較大的補充資料集。預設 mapping 會把 `Miscellaneous Trash` 與
`Textile Trash` 視為弱 `accept`，其他 material classes 視為弱 `reject`；這是為了先追
accuracy/accept recall，正式驗收前仍需抽樣審閱。

```bash
mkdir -p data/sources/realwaste
curl -L --fail \
  --output data/sources/realwaste/realwaste.zip \
  https://cdn.uci-ics-mlr-prod.aws.uci.edu/908/realwaste.zip
unzip -q data/sources/realwaste/realwaste.zip -d data/sources/realwaste/extracted

python3 scripts/build-folder-yolo-cls-dataset.py \
  --source-dir data/sources/realwaste/extracted/realwaste-main/RealWaste \
  --output-dir data/training/realwaste_yolo_cls \
  --mapping-preset realwaste \
  --max-train-majority-ratio 2.0

uv run --with ultralytics --with torch --with torchvision \
  --with onnx --with onnxruntime --with onnxslim \
  python scripts/train-yolo-cls.py \
  --data data/training/realwaste_yolo_cls \
  --model exports/20260601T122805Z/best.pt \
  --project runs/realwaste-accuracy \
  --serial-prefix realwaste-accuracy \
  --epochs 30 \
  --batch 32 \
  --device 0 \
  --export-onnx
```
