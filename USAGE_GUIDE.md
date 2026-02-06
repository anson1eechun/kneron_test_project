# 🚀 本地推論使用指南

> 快速開始：在本機上運行訓練好的模型

---

## ✅ 測試成功！

您的模型已經可以正常使用了！測試結果：

```
預測結果: Ant (螞蟻)
置信度: 69.17%
```

---

## 📋 使用方法

### 方法 1：使用簡單測試腳本（推薦，最簡單）

```powershell
# 測試單張圖片
py test_inference_simple.py
```

**優點**：
- ✅ 無需參數，直接運行
- ✅ 自動使用預設的測試圖片
- ✅ 顯示詳細的推論結果

**輸出範例**：
```
============================================================
本地推論測試
============================================================

📦 正在載入模型: ants_bees_opt_fixed.onnx
✅ 模型載入成功
   輸入名稱: input
   輸入形狀: [1, 3, 224, 224]

🖼️  正在讀取圖片: data/val/ants/10308379_1b6c72e180.jpg
✅ 圖片預處理完成
   圖片形狀: (1, 3, 224, 224)

🔍 開始推論...

============================================================
推論結果
============================================================
原始輸出: [ 0.74049133 -0.06772483]

概率分佈:
  Ant (螞蟻): 69.17% ←
  Bee (蜜蜂): 30.83%

預測結果: Ant (螞蟻)
置信度: 69.17%
============================================================

✅ 測試完成！
```

---

### 方法 2：使用改進版推論工具（功能更豐富）

```powershell
# 單張圖片推論
py inference_local.py -i data/val/ants/10308379_1b6c72e180.jpg

# 使用不同的模型
py inference_local.py -i image.jpg -m ants_bees.onnx

# 批量處理資料夾
py inference_local.py -i data/val/ants/ --batch

# 簡潔輸出模式
py inference_local.py -i image.jpg --quiet
```

---

### 方法 3：使用原始測試腳本

```powershell
py inference_test_local.py
```

---

## 🔧 修改測試圖片

### 修改 `test_inference_simple.py`

編輯腳本中的這一行：

```python
TEST_IMAGE = "data/val/ants/10308379_1b6c72e180.jpg"  # 改成您的圖片路徑
```

然後運行：

```powershell
py test_inference_simple.py
```

---

## 📊 可用的模型文件

您可以使用以下任一模型：

| 模型文件 | 說明 | 推薦度 |
|---------|------|--------|
| `ants_bees.onnx` | 原始導出的模型 | ⭐⭐ |
| `ants_bees_opt.onnx` | Kneron 優化後的模型 | ⭐⭐⭐ |
| `ants_bees_opt_fixed.onnx` | 修復 ReduceMean 後的模型 | ⭐⭐⭐⭐ **推薦** |

預設使用 `ants_bees_opt_fixed.onnx`。

---

## 🎯 快速測試不同圖片

### 測試螞蟻圖片

```powershell
# 修改 test_inference_simple.py 中的 TEST_IMAGE
# 或直接使用 inference_local.py
py inference_local.py -i data/val/ants/您的圖片.jpg
```

### 測試蜜蜂圖片

```powershell
py inference_local.py -i data/val/bees/您的圖片.jpg
```

---

## 📝 在 Python 代碼中使用

```python
import onnxruntime as rt
import numpy as np
from PIL import Image

# 載入模型
session = rt.InferenceSession('ants_bees_opt_fixed.onnx')
input_name = session.get_inputs()[0].name

# 預處理圖片
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    img_data = np.array(img).astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std
    img_data = img_data.astype('float32')
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

# 執行推論
input_data = preprocess('your_image.jpg')
output = session.run(None, {input_name: input_data})

# 解析結果
raw_result = output[0][0]
predicted_idx = np.argmax(raw_result)
classes = ['Ant (螞蟻)', 'Bee (蜜蜂)']
print(f"預測結果: {classes[predicted_idx]}")
```

---

## ⚠️ 注意事項

1. **使用 `py` 命令**：在 Windows PowerShell 中，使用 `py` 而非 `python`
2. **編碼問題**：腳本已修復 Windows 編碼問題，應該可以正常顯示中文和 emoji
3. **模型路徑**：確保模型文件在當前目錄或使用絕對路徑

---

## 🐛 問題排查

### 問題：找不到模型文件

**解決**：
1. 確認模型文件存在
2. 使用絕對路徑：
   ```python
   MODEL_PATH = "G:/workplace/kneron_project/ants_bees_opt_fixed.onnx"
   ```

### 問題：找不到圖片文件

**解決**：
1. 確認圖片路徑正確
2. 使用絕對路徑
3. 檢查文件擴展名（.jpg, .jpeg, .png）

### 問題：依賴缺失

**解決**：
```powershell
py -m pip install onnxruntime pillow numpy
```

---

## 📚 相關文件

- `test_inference_simple.py` - 簡單測試腳本（推薦）
- `inference_local.py` - 功能完整的推論工具
- `inference_test_local.py` - 原始測試腳本
- `LOCAL_INFERENCE_GUIDE.md` - 詳細使用指南

---

**最後更新**: 2024-02-07  
**狀態**: ✅ 測試成功，可以正常使用

