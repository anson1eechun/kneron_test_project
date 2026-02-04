# Part-06 軟體模擬推論說明

## 測試結果

### ✅ 本地測試成功

在 Windows 本地環境中成功執行了推論測試：

**測試結果：**
- **Test 1 - 螞蟻圖片**：
  - Raw Output: [ 1.5317764 -1.6757964]
  - 預測結果: **Ant (螞蟻)** ✓
  - 第一個數值（1.53）大於第二個數值（-1.68），正確預測為螞蟻

- **Test 2 - 蜜蜂圖片**：
  - Raw Output: [-1.3074203  1.2468289]
  - 預測結果: **Bee (蜜蜂)** ✓
  - 第二個數值（1.25）大於第一個數值（-1.31），正確預測為蜜蜂

### ✅ Docker 容器測試成功

升級 `onnxruntime` 後，Docker 容器中的測試也成功通過：

**執行方式：**
```bash
docker run --rm -v ${PWD}:/docker_mount kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && pip install --upgrade onnxruntime && python /docker_mount/inference_test.py"
```

**測試結果：**
- **Test 1 - 螞蟻圖片**：預測結果: **Ant (螞蟻)** ✓
- **Test 2 - 蜜蜂圖片**：預測結果: **Bee (蜜蜂)** ✓

## 測試方式

### 方式 1：Windows 本地測試（推薦，快速）

```bash
py inference_test_local.py
```

### 方式 2：Docker 容器測試

```bash
docker run --rm -v ${PWD}:/docker_mount kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && pip install --upgrade onnxruntime && python /docker_mount/inference_test.py"
```

## 結論

✅ **模型驗證通過**：模型能夠正確識別螞蟻和蜜蜂，功能正常。

可以繼續進行：
- **Part-07**：編譯成硬體格式

## 測試腳本

- `inference_test_local.py` - Windows 本地測試版本（已驗證可用）
- `inference_test.py` - Docker 容器測試版本（需要更新 onnxruntime）

