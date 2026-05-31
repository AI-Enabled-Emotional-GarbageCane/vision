# Vision v0.3 規格

本文件是 `vision` repo 的可實作與可驗收規格。中心 repo 仍是跨 repo public contract 的來源；本 repo 以 `contracts/contract.lock.json` 鎖定 v0.3。

## 職責與邊界

| 項目 | 規格 |
|---|---|
| Owns | L515 RGB capture、YOLOv11n binary classification、`recognition_result` |
| Consumes | `q_detected` 內的 `user_detected` |
| Produces | `q_result` 內的 `recognition_result` |
| Must not own | UI、語音播放、LED、L515 depth 距離觸發 |
| v1 限制 | 不做多物件偵測，固定輸出 `num_objects=1` |

## 資料規格

- 公開資料以 TrashNet + RealWaste 為主；TACO 不列入 v1。
- 另用 Intel RealSense L515 於 demo 桶口角度補拍每類 50-100 張，作為場景校正與最終驗收集。
- Binary label mapping：
  - `accept`：一般垃圾 / 可燃垃圾，本垃圾桶接受的投入物。
  - `reject`：紙、塑膠、金屬、玻璃、廚餘等不應投入本垃圾桶的物件。
- 資料切分需保留獨立 test set；報告最低需列出 test set top-1 accuracy。

## 訓練規格

- 模型固定為 YOLOv11n classification，預設起點為 Ultralytics `yolo11n-cls.pt`。
- 標準訓練入口：

```bash
yolo classify train data=<dataset-dir> model=yolo11n-cls.pt epochs=<n> imgsz=<size>
```

- 訓練完成後輸出 PyTorch 權重，再匯出 ONNX：

```bash
yolo export model=<best.pt> format=onnx
```

- Jetson 階段再由 ONNX / PyTorch 權重轉 TensorRT engine；真權重、ONNX、engine 不進 git。

## 推論與 Payload

收到 `user_detected` 後，`vision` 擷取 L515 RGB frame，執行 YOLOv11n classification，保存一張快照，並送出：

```json
{
  "event": "recognition_result",
  "class": "accept",
  "confidence": 0.91,
  "num_objects": 1,
  "snapshot_path": "snapshots/example.jpg",
  "ts": "2026-05-31T20:00:00"
}
```

- `class` 只能是 `accept` 或 `reject`。
- `confidence` 必須在 0 到 1 之間。
- `confidence < 0.5` 時仍送最佳猜測的 `class`；由 display 依低信心播放「看不出來」回饋，不採用該 class 做 accept/reject 判定。
- v1 固定 `num_objects=1`；`num_objects > 1` 規則保留給未來 detection / foreground estimation 版本。

## 部署與 Artifact Policy

- 順序：先以筆電 + L515/OpenCV 跑通 PoC，再移植 Jetson AGX Orin Nano + TensorRT。
- 允許進 git：小型 sample/mock fixture、設定檔、文件、contract lock、驗證腳本。
- 不進 git：`data/`、`runs/`、`weights/`、`exports/`、`.pt`、`.onnx`、`.engine`。

## 驗收標準

| 類別 | 最低標準 |
|---|---|
| Contract | `recognition_result` 欄位完整，`class` 只允許 `accept` / `reject`。 |
| Smoke test | `./validate.sh` 必跑 stub model smoke test，確認 pipeline 可產生合法 payload。 |
| 模型準確率 | Test set top-1 accuracy >= 85%。 |
| 部署 | 筆電 PoC 先跑通；Jetson 端再測 TensorRT latency。 |
| Snapshot | 每次推論輸出可被 display 記錄的本機 `snapshot_path`。 |

## 目前結論

v0.3 不改 queue、event 或 required payload fields；主要更新為 L515、YOLOv11n classification、`num_objects=1` v1 限制，以及可在無真模型權重時執行的 stub smoke test。
