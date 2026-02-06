# Part-07 編譯狀態更新

## 當前進度

### ✅ 已完成的準備工作
1. 校準圖片準備完成（30 張圖片）
2. 配置文件已建立並修正格式
3. 創建了定點分析腳本 `run_fp_analysis.py`
4. 合併了 ONNX 外部數據文件為單一文件

### ⚠️ 遇到的問題

#### 問題 1：ONNX 外部數據文件
- **問題**：原始 ONNX 模型使用了外部數據文件格式（`ants_bees.onnx.data`）
- **解決**：已創建 `ants_bees_merged.onnx` 單一文件（90MB）

#### 問題 2：定點分析執行錯誤
- **進度**：定點分析進度達到 100%
- **錯誤**：在最後階段出現斷言錯誤：
  ```
  Assertion `weight_radix_vect.size() == (size_t)o_c` failed
  ```
- **可能原因**：
  - 模型結構與工具鏈不完全兼容
  - ResNet50 的某些層結構需要特殊處理
  - 需要調整量化參數

## 下一步建議

### 方案 1：檢查生成的 .bie 文件
即使出現錯誤，工具鏈可能已經生成了部分結果。需要檢查：
- `/data1/fpAnalyser/` 目錄
- `/workspace/.tmp/` 目錄

### 方案 2：調整量化參數
嘗試修改 `input_params.json` 中的參數：
- 調整 `radix` 值
- 修改 `outlier` 參數
- 嘗試不同的 `quantize_mode`

### 方案 3：使用優化後的模型
嘗試使用經過 Kneron 優化工具處理的模型（如果可用）

### 方案 4：聯繫 Kneron 支持
如果問題持續，可能需要：
- 確認工具鏈版本與模型兼容性
- 獲取 ResNet50 的特定配置建議

## 當前文件狀態

- ✅ `input_params.json` - 已配置（使用 kneron 預處理方法）
- ✅ `batch_input_params.json` - 已配置
- ✅ `ants_bees_merged.onnx` - 已生成（90MB，單一文件）
- ✅ `run_fp_analysis.py` - 定點分析腳本
- ⏳ `.bie` 文件 - 待確認是否生成

## 技術細節

### 使用的預處理方法
從 `normalize` 改為 `kneron`，因為：
- Kneron 工具鏈期望使用其專用的預處理方法
- `normalize` 方法在圖片轉換時出現問題

### 模型文件
- 原始：`ants_bees.onnx` + `ants_bees.onnx.data`（外部數據）
- 合併後：`ants_bees_merged.onnx`（單一文件，90MB）

