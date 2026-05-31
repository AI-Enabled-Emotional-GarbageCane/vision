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

預設驗證只跑 contract/spec 檢查與 stub model smoke test；真模型權重與 TensorRT engine 不進 git，也不作為基本驗證門檻。
