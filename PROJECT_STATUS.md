# Kneron ResNet50 專案現狀報告

## ✅ 已完成的工作

### 1. 模型訓練（Part-04）✅
- **狀態**：已完成
- **模型架構**：ResNet50
- **訓練準確度**：96.1%
- **驗證準確度**：96.1%
- **訓練時間**：約 2 分 22 秒
- **輸出文件**：`ants_bees.onnx`（235KB + 90MB 外部數據）

### 2. ONNX 模型導出 ✅
- **狀態**：已完成
- **模型文件**：
  - `ants_bees.onnx` - 原始 ONNX 模型（使用外部數據）
  - `ants_bees_merged.onnx` - 合併後的單一文件（90MB）
  - `ants_bees_opt.onnx` - 優化版本（目前為原始模型的副本）

### 3. 推論測試（Part-06）✅
- **狀態**：測試成功
- **本地測試**：✓ 通過
- **Docker 測試**：✓ 通過（升級 onnxruntime 後）
- **測試結果**：
  - 螞蟻圖片：正確預測為 Ant ✓
  - 蜜蜂圖片：正確預測為 Bee ✓

## ⚠️ 遇到的問題

### Part-05：ONNX 模型優化

#### 問題 1：版本兼容性
- **問題**：Kneron 工具鏈中的 `onnx2onnx.py` 依賴舊版 ONNX API
- **錯誤**：`ImportError: cannot import name 'optimizer' from 'onnx'`
- **狀態**：未解決

#### 問題 2：使用 ktc 優化工具
根據 PDF 文檔（Part_05_Onnx檔案轉換(Convert)_ok8.pdf），嘗試使用 `ktc.onnx_optimizer.onnx2onnx_flow()` 進行優化，但遇到：

1. **IR 版本問題**：
   - 原始模型 IR 版本：10
   - 工具鏈支持：IR 版本 ≤ 6
   - 嘗試降級後出現節點兼容性問題

2. **Opset 版本問題**：
   - 原始模型 Opset：18
   - 降級到 Opset 11 後，`ReduceMean` 節點出現兼容性問題
   - 錯誤：`Node (node_mean) has input size 2 not in range [min=1, max=1]`

### Part-07：模型編譯

#### 問題：定點分析錯誤
- **進度**：定點分析執行到 100%
- **錯誤**：`Assertion weight_radix_vect.size() == (size_t)o_c failed`
- **狀態**：未完成，未生成 .bie 文件

## 📊 專案文件清單

### 模型文件
- ✅ `ants_bees.onnx` - 原始 ONNX 模型（235KB + 外部數據）
- ✅ `ants_bees_merged.onnx` - 合併後的單一文件（90MB）
- ⚠️ `ants_bees_opt.onnx` - 優化版本（目前為副本，未真正優化）

### 配置和腳本
- ✅ `input_params.json` - 輸入配置（已調整為 Kneron 格式）
- ✅ `batch_input_params.json` - 批次編譯配置
- ✅ `ants_bees_convert.py` - 根據 PDF 文檔創建的優化腳本
- ✅ `run_fp_analysis.py` - 定點分析腳本
- ✅ `inference_test.py` - Docker 推論測試腳本
- ✅ `inference_test_local.py` - 本地推論測試腳本

### 數據文件
- ✅ `data/train/` - 訓練數據（244 張圖片）
- ✅ `data/val/` - 驗證數據（153 張圖片）
- ✅ `imgs/` - 校準圖片（30 張，用於編譯）

## 🔍 根本原因分析

### 主要問題
1. **ONNX 版本不匹配**：
   - PyTorch 2.10.0 導出的 ONNX 模型使用 IR 版本 10 和 Opset 18
   - Kneron 工具鏈（onnx1.13 環境）只支持 IR 版本 ≤ 6 和 Opset ≤ 11

2. **模型結構兼容性**：
   - ResNet50 的某些操作（如 ReduceMean）在新版 ONNX 中的實現與舊版不兼容
   - 直接降級會導致節點驗證失敗

## 💡 解決方案建議

### 方案 1：使用較舊版本的 PyTorch 導出 ONNX（推薦）
```python
# 使用 PyTorch 1.x 或較舊版本導出 ONNX
# 確保使用 opset_version=11 或更低
torch.onnx.export(
    model_ft,
    dummy_input,
    "ants_bees.onnx",
    opset_version=11,  # 使用較低的 opset 版本
    ...
)
```

### 方案 2：使用 ONNX 版本轉換工具
- 使用 `onnx-simplifier` 或其他工具進行版本轉換
- 可能需要手動修復某些不兼容的節點

### 方案 3：聯繫 Kneron 支持
- 獲取更新的工具鏈版本
- 獲取 ResNet50 的特定配置建議
- 確認是否有針對新版 ONNX 的解決方案

### 方案 4：跳過優化步驟
- 直接使用原始 ONNX 模型進行編譯
- 某些情況下，編譯器可能能夠處理未優化的模型

## 📝 當前狀態總結

| 階段 | 狀態 | 備註 |
|------|------|------|
| Part-04: 模型訓練 | ✅ 完成 | 準確度 96.1% |
| Part-04: ONNX 導出 | ✅ 完成 | 已生成 ONNX 文件 |
| Part-05: ONNX 優化 | ⚠️ 部分完成 | 遇到版本兼容性問題 |
| Part-06: 推論測試 | ✅ 完成 | 測試通過 |
| Part-07: 模型編譯 | ⚠️ 未完成 | 定點分析階段失敗 |

## 🎯 結論

**模型已經訓練完成並可以正常使用**：
- ✅ 訓練準確度：96.1%
- ✅ 推論測試：成功
- ✅ ONNX 模型：已導出並驗證

**ONNX 優化遇到技術障礙**：
- ⚠️ 版本兼容性問題（ONNX IR 10 vs 工具鏈支持 ≤ 6）
- ⚠️ 需要調整導出參數或使用兼容的工具鏈版本

**建議**：
1. 優先嘗試使用較舊版本的 PyTorch 重新導出 ONNX（opset_version=11）
2. 或聯繫 Kneron 支持獲取更新的工具鏈或配置建議
3. 當前模型已經可以正常進行推論，優化步驟可以暫時跳過

