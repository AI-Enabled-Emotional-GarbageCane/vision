# 一般垃圾 positive fine-tuning 報告

日期：2026-06-02

## 目的

補強目前模型對「一般垃圾 positive」的接受能力，重點類別包含菸蒂、衛生紙/餐巾紙、垃圾袋、髒包裝、小型不可回收雜物。所有新增標籤目前都屬於弱標籤，正式指標仍需要人工審閱後再定案。

## 新增資料

資料來源：

- TACO: https://github.com/pedropro/TACO
- TACO annotations: https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json
- TIDY: https://github.com/gale31/TIDY

TACO accept-focus 資料集：

- 路徑：`data/inference_general_trash_positive/taco_full_accept_focus`
- 總數：1451
- accept_weak：783
- reject_weak：668
- accept 類別：Cigarette 154、Tissues 24、Garbage bag 13、Other plastic wrapper 146、Plastic film 242、Single-use carrier bag 42、Crisp packet 28、Disposable food container 30、Foam food container 12、Plastic utensils 25、Styrofoam piece 67

TIDY 資料集：

- 原始路徑：`data/sources/TIDY`
- YOLO cls 路徑：`data/training/tidy_general_trash_yolo_cls`
- 總數：284
- accept：cigarette-butt、plastic-bag、plastic、unknown
- reject：glass、metal、paper、plastic-bottle

合併訓練資料：

- `data/training/general_trash_positive_combined_yolo_cls`：4868 張，train 1247/1675、val 267/706、test 267/706
- `data/training/general_trash_positive_relaxed_yolo_cls`：4048 張，train 1247/930、val 267/669、test 267/668

## 訓練輸出

第一輪：

- run：`runs/general-trash-positive/general-trash-positive-001`
- base：`runs/realwaste-accuracy/realwaste-accuracy-002/weights/best.pt`
- best：`runs/general-trash-positive/general-trash-positive-001/weights/best.pt`
- onnx：`runs/general-trash-positive/general-trash-positive-001/weights/best.onnx`

第二輪：

- run：`runs/general-trash-positive/general-trash-positive-002`
- base：`runs/general-trash-positive/general-trash-positive-001/weights/best.pt`
- best：`runs/general-trash-positive/general-trash-positive-002/weights/best.pt`
- onnx：`runs/general-trash-positive/general-trash-positive-002/weights/best.onnx`

## 弱標籤測試結果

| model | test set | weak agreement | gate accept recall | reject false accept |
| --- | --- | ---: | ---: | ---: |
| 001 | combined test | 82.84% | 34.08% | 1.70% |
| 001 | TACO full | 60.30% | 8.94% | 3.89% |
| 001 | TIDY test | 64.29% | 37.04% | 0.00% |
| 002 | combined test | 81.09% | 51.69% | 6.09% |
| 002 | relaxed test | 83.53% | 51.69% | 5.54% |
| 002 | TACO full | 59.96% | 26.56% | 11.68% |
| 002 | TIDY test | 57.14% | 48.15% | 13.33% |
| 002 | RealWaste full test | 91.30% | 81.97% | 4.40% |
| 002 | TACO reject-safety | 52.22% | 20.00% | 7.50% |
| 002 | Taiwan mapped weak set | 56.00% | 24.00% | 4.00% |

## 判斷

`general-trash-positive-002` 確實比原本模型和第一輪更願意接受一般垃圾 positive，尤其 TACO full 的 gate accept recall 從第一輪 8.94% 提升到 26.56%，combined/relaxed test 達到 51.69%。但這不是可直接部署的安全模型，因為 reject false accept 同時升高，TACO full 到 11.68%，TIDY test 到 13.33%，RealWaste full test 也從先前最佳模型的約 95.79% agreement 降到 91.30%。

目前主要錯誤集中在 TACO reject 類別被吸進 accept，例如 clear plastic bottle、drink can、other plastic、disposable plastic cup、plastic straw、plastic bottle cap。這表示單純用整張圖分類 fine-tune，會把「塑膠/小物/街景垃圾」這種外觀特徵學成 accept，而不是真的學會台灣規則下的一般垃圾邊界。

## 建議

暫時保留 `general-trash-positive-002` 作為實驗權重，不建議取代目前最佳安全權重。下一輪應改成三件事：

1. 對 TACO/TIDY 弱標籤做人工抽樣審閱，先清掉最容易錯的 reject 類別，尤其寶特瓶、飲料罐、瓶蓋、紙類、玻璃。
2. 改成物件 crop 或偵測後再分類，不要只用整張圖分類；菸蒂、衛生紙、小包裝在大場景裡太小，整圖 classifier 很容易學到背景。
3. 分開優化「accept recall」和「reject safety」：一般垃圾 positive 可以用較寬的候選模型，但最後 gate 仍應保留安全模型或提高 threshold，避免回收物被高信心 accept。

## 驗證

已執行：

- `python3 -m py_compile scripts/*.py tests/*.py`
- `./validate.sh`

結果：全部通過。
