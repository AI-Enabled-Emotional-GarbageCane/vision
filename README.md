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

## 模型紀錄

- 目前所有 curated exports 的紀錄見 [`docs/model-registry.md`](./docs/model-registry.md)。
- AGX + L515 實機串接與糖果包裝塑膠紙模型問題見 [`docs/agx-l515-vision-integration-20260604.md`](./docs/agx-l515-vision-integration-20260604.md)。
