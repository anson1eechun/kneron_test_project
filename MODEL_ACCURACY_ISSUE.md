# ⚠️ 模型精度問題分析報告

> 發現優化過程導致模型精度嚴重下降

---

## 📊 測試結果總結

### 準確度對比

| 模型 | 準確度 | 平均置信度差異 | 狀態 |
|------|--------|---------------|------|
| **原始模型** (`ants_bees.onnx`) | **100%** (2/2) | **82.34%** | ✅ **完美** |
| 優化模型 (`ants_bees_opt.onnx`) | 50% (1/2) | 27.07% | ❌ **嚴重問題** |
| 修復模型 (`ants_bees_opt_fixed.onnx`) | 50% (1/2) | 27.07% | ❌ **嚴重問題** |

---

## 🔍 詳細測試結果

### 原始模型 (`ants_bees.onnx`) ✅

**螞蟻圖片測試**：
- 原始輸出: `[ 0.94412917 -1.1827531 ]`
- 概率: Ant 89.35% vs Bee 10.65%
- 預測: ✅ Ant (螞蟻)
- 置信度差異: **78.70%**

**蜜蜂圖片測試**：
- 原始輸出: `[-1.4134778  1.171571 ]`
- 概率: Ant 7.01% vs Bee 92.99%
- 預測: ✅ Bee (蜜蜂)
- 置信度差異: **85.98%**

**結論**：模型表現完美，高置信度，準確無誤。

---

### 優化模型 (`ants_bees_opt.onnx`) ❌

**螞蟻圖片測試**：
- 原始輸出: `[ 0.7404913  -0.06772488]`
- 概率: Ant 69.17% vs Bee 30.83%
- 預測: ✅ Ant (螞蟻) - **正確但置信度下降**
- 置信度差異: **38.35%**（下降 40%）

**蜜蜂圖片測試**：
- 原始輸出: `[0.43779117 0.11916406]`
- 概率: Ant 57.90% vs Bee 42.10%
- 預測: ❌ **Ant (螞蟻) - 錯誤！應該是蜜蜂**
- 置信度差異: **15.80%**（非常不確定）

**結論**：優化過程導致：
1. 精度損失（蜜蜂被誤判為螞蟻）
2. 置信度大幅下降（從 85.98% 降到 15.80%）
3. 模型變得不確定

---

### 修復模型 (`ants_bees_opt_fixed.onnx`) ❌

**結果**：與優化模型完全相同

**結論**：ReduceMean 的替換沒有改善問題，問題根源在優化過程本身。

---

## 🔎 問題根源分析

### 1. 優化過程的問題

`ants_bees_opt.onnx` 是通過 `ktc.onnx_optimizer.onnx2onnx_flow()` 生成的：

```python
onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(
    onnx_model,
    eliminate_tail=False
)
```

**可能的原因**：
1. **數值精度損失**：優化過程可能改變了浮點數精度
2. **操作融合問題**：Batch Normalization 融合可能不正確
3. **圖結構改變**：優化可能改變了計算圖的結構
4. **權重量化**：優化過程可能隱式地進行了量化

### 2. ReduceMean 替換的問題

`ants_bees_opt_fixed.onnx` 將 ReduceMean 替換為 GlobalAveragePool：

**觀察**：
- 替換後結果與優化模型完全相同
- 說明問題不在 ReduceMean，而在優化過程

---

## 💡 解決方案

### 方案 1：使用原始模型（推薦）✅

**直接使用 `ants_bees.onnx` 進行部署**：

```python
# 本地推論
MODEL_PATH = "ants_bees.onnx"

# Kneron 編譯
config = ktc.ModelConfig(
    id=100,
    version="0000",
    platform="520",
    onnx_path="ants_bees.onnx"  # 使用原始模型
)
nef_path = ktc.compile([config])
```

**優點**：
- ✅ 100% 準確度
- ✅ 高置信度（82.34% 平均差異）
- ✅ 無精度損失

**缺點**：
- ⚠️ 可能需要處理 ReduceMean 問題（如果編譯時遇到）

---

### 方案 2：重新導出並優化

**步驟**：
1. 從 PyTorch 重新導出 ONNX（確保使用正確的參數）
2. 跳過 Kneron 優化，直接編譯
3. 如果編譯失敗，再進行最小化的操作替換

```python
# 重新導出
torch.onnx.export(
    model,
    dummy_input,
    "ants_bees_v2.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    export_params=True,
    do_constant_folding=True,
    dynamic_axes=None
)

# 直接編譯，不進行優化
config = ktc.ModelConfig(
    id=100,
    version="0000",
    platform="520",
    onnx_path="ants_bees_v2.onnx"
)
```

---

### 方案 3：檢查優化參數

**嘗試不同的優化參數**：

```python
# 嘗試不同的優化選項
onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(
    onnx_model,
    eliminate_tail=False,
    # 可能需要其他參數來保持精度
)
```

**注意**：需要查閱 Kneron Toolchain 文檔，了解所有可用的優化參數。

---

### 方案 4：驗證優化後的模型

**在優化後立即驗證**：

```python
# 優化後立即測試
import onnxruntime as rt

original_session = rt.InferenceSession("ants_bees.onnx")
optimized_session = rt.InferenceSession("ants_bees_opt.onnx")

# 使用相同的輸入測試
# 比較輸出差異
```

---

## 📋 建議行動計劃

### 立即行動

1. **✅ 使用原始模型進行部署**
   - 本地推論：使用 `ants_bees.onnx`
   - Kneron 編譯：直接編譯 `ants_bees.onnx`

2. **⚠️ 如果編譯失敗**
   - 只進行必要的操作替換（如 ReduceMean → GlobalAveragePool）
   - 不要進行完整的優化流程

3. **🔍 調查優化問題**
   - 聯繫 Kneron 支持，報告優化導致的精度損失
   - 詢問是否有保持精度的優化參數

### 長期改進

1. **建立驗證流程**
   - 每次優化後立即測試準確度
   - 建立自動化測試腳本

2. **文檔記錄**
   - 記錄哪些優化步驟會影響精度
   - 建立最佳實踐指南

---

## 🎯 結論

**關鍵發現**：
1. ✅ **原始模型 (`ants_bees.onnx`) 表現完美**，應該直接使用
2. ❌ **Kneron 優化過程導致嚴重精度損失**，不應該使用
3. ⚠️ **ReduceMean 替換不是問題根源**，問題在優化過程

**建議**：
- **立即使用 `ants_bees.onnx` 進行所有後續工作**
- **跳過優化步驟，直接編譯原始模型**
- **如果編譯時遇到 ReduceMean 問題，只進行該操作的替換**

---

**最後更新**: 2024-02-07  
**狀態**: ⚠️ 發現嚴重問題，已提供解決方案

