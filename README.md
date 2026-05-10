# vision

AI 影像辨識服務 — [AI 情緒垃圾筒](https://github.com/AI-Enabled-Emotional-GarbageCane/ai-enabled-emotional-garbagecane) 的視覺子系統。

## 職責

- 影像資料集收集與標註
- 模型訓練(垃圾分類)
- 推論 service:讀取攝影機畫面 → 輸出 `{class, confidence}`

## 介面

- 接收 firmware 的觸發訊號(`user_detected`)後啟動推論
- 將結果(`recognition_result`)發送給 display

詳細 API 契約見 [monorepo `docs/api-contract.md`](https://github.com/AI-Enabled-Emotional-GarbageCane/ai-enabled-emotional-garbagecane/blob/main/docs/api-contract.md)。

## 技術棧

TBD
