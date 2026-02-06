# ⚡ Kneron 開發快速參考指南

> 快速查閱常見問題和解決方案

---

## 🚨 常見錯誤快速修復

### 1. ONNX 導出問題

**錯誤**: `Setting ONNX exporter to use operator set version 18...`

**解決**:
```python
torch.onnx.export(..., opset_version=11)  # 明確指定版本
```

---

### 2. ONNX Runtime 版本不兼容

**錯誤**: `Unsupported model IR version: 10, max supported IR version: 9`

**解決**:
```bash
# Docker 容器內
pip install --upgrade onnxruntime
```

---

### 3. 數據類型不匹配

**錯誤**: `Unexpected input data type. Actual: (tensor(double)), expected: (tensor(float))`

**解決**:
```python
img_data = np.array(img).astype('float32')  # 明確指定 float32
```

---

### 4. onnx2onnx.py 無法使用

**錯誤**: `ImportError: cannot import name 'optimizer' from 'onnx'`

**解決**:
```python
# 使用 ktc API 替代
import ktc
onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(onnx_model)
```

---

### 5. ReduceMean 不支持

**錯誤**: `UnimplementedFeature: undefined CPU op [ReduceMean]`

**解決**:
```python
# 使用 fix_reducemean_properly.py 替換為 GlobalAveragePool
python fix_reducemean_properly.py
```

---

### 6. 定點分析失敗

**錯誤**: `Assertion weight_radix_vect.size() == (size_t)o_c failed`

**解決**:
```python
# 跳過定點分析，直接編譯
config = ktc.ModelConfig(id=100, version="0000", platform="520", onnx_path="model.onnx")
nef_path = ktc.compile([config])
```

---

### 7. batchCompile 只支持 .bie

**錯誤**: `Currently, batch compile only support models after fix point analysis`

**解決**:
```python
# 使用 ktc.compile() API，支持直接編譯 ONNX
nef_path = ktc.compile([config])
```

---

## 📋 標準工作流程

### 步驟 1: 導出 ONNX
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=11,  # 關鍵！
    input_names=['input'],
    output_names=['output']
)
```

### 步驟 2: 優化 ONNX
```python
import ktc
onnx_model = onnx.load("model.onnx")
onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(onnx_model)
onnx.save(onnx_opt, "model_opt.onnx")
```

### 步驟 3: 修復不支持的操作
```python
python fix_reducemean_properly.py
```

### 步驟 4: 編譯
```python
config = ktc.ModelConfig(
    id=100, version="0000", platform="520",
    onnx_path="model_opt_fixed.onnx"
)
nef_path = ktc.compile([config])
```

---

## 🔧 工具版本要求

| 工具 | 推薦版本 | 注意事項 |
|------|---------|---------|
| ONNX Opset | 11 | 與 Kneron 兼容 |
| ONNX IR | 6-9 | Docker 容器限制 |
| 數據類型 | float32 | 明確指定 |
| Kneron Toolchain | latest | 使用 ktc API |

---

## 📁 關鍵文件

- `fix_onnx_export.py` - 導出兼容 ONNX
- `complete_optimization.py` - ONNX 優化
- `fix_reducemean_properly.py` - 修復 ReduceMean
- `direct_compile.py` - 直接編譯

---

## 💡 最佳實踐

1. ✅ **明確指定版本**: `opset_version=11`
2. ✅ **明確數據類型**: `.astype('float32')`
3. ✅ **使用 API 而非腳本**: `ktc.compile()` 而非 `batchCompile_520.py`
4. ✅ **檢查操作兼容性**: 替換不支持的操作
5. ✅ **充分測試**: 推論測試驗證功能

---

**詳細文檔**: 參見 `DEVELOPMENT_EXPERIENCE.md`

