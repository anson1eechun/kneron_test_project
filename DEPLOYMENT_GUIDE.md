# AI Dongle 部署完整指南

## 目標
完成整個流程並部署到 Kneron AI Dongle

## 當前狀態

✅ **已完成**：
- Part-04: 模型訓練（準確度 96.1%）
- Part-06: 推論測試（測試通過）

⚠️ **需要解決**：
- Part-05: ONNX 優化（版本兼容性問題）
- Part-07: 模型編譯（生成 .nef 文件）
- Part-08: AI Dongle 部署

## 解決方案：三步驟

### 步驟 1：重新導出兼容的 ONNX 模型

**問題**：當前 ONNX 模型使用 IR 版本 10，工具鏈只支持 ≤ 6

**解決方法**：

```bash
# 在 Windows 本地執行
py fix_onnx_export.py
```

這會生成 `ants_bees_compatible.onnx`，使用 opset_version=11。

### 步驟 2：在 Docker 中優化 ONNX

**根據 Part-05 PDF 文檔**：

```bash
# 啟動 Docker 容器
docker run --rm -it -v ${PWD}:/docker_mount kneron/toolchain:latest

# 在容器內執行
cd /data1
cp /docker_mount/ants_bees_compatible.onnx .
cp /docker_mount/complete_optimization.py .

# 執行優化
source /workspace/miniconda/bin/activate onnx1.13
python complete_optimization.py
```

這會生成 `ants_bees_opt.onnx`。

### 步驟 3：編譯生成 .nef 文件

**根據 Part-07 PDF 文檔**：

#### 3.1 準備校準圖片
- ✅ 已完成：`imgs/` 資料夾有 30 張圖片

#### 3.2 更新配置文件

更新 `input_params.json`：
```json
{
    "model_info": {
        "input_onnx_file": "ants_bees_opt.onnx",
        "model_inputs": [{
            "model_input_name": "input",
            "input_image_folder": "imgs"
        }],
        "quantize_mode": "default",
        "outlier": 0.999
    },
    "preprocess": {
        "img_preprocess_method": "kneron",
        "img_channel": "RGB",
        "radix": 8,
        "keep_aspect_ratio": false,
        "pad_mode": 1,
        "p_crop": {
            "crop_x": 0,
            "crop_y": 0,
            "crop_w": 0,
            "crop_h": 0
        }
    }
}
```

更新 `batch_input_params.json`：
```json
{
    "models": [
        {
            "id": 100,
            "version": "0000",
            "path": "ants_bees_opt.onnx",
            "input_params": "input_params.json"
        }
    ],
    "thread_num": 1,
    "version_check": false
}
```

#### 3.3 執行定點分析和編譯

```bash
# 在 Docker 容器內
cd /docker_mount

# 執行定點分析
source /workspace/miniconda/bin/activate onnx1.13
python run_fp_analysis.py

# 如果成功生成 .bie 文件，更新 batch_input_params.json
# 將 path 改為生成的 .bie 文件名

# 執行編譯
python /workspace/scripts/batchCompile_520.py -c batch_input_params.json
```

### 步驟 4：部署到 AI Dongle（Part-08）

**根據 Part-08 PDF 文檔**：

1. 將生成的 `models_520.nef` 文件複製到 AI Dongle
2. 使用 Kneron SDK 進行推論

## 故障排除

### 如果步驟 2 失敗（ONNX 優化）

**選項 A**：跳過優化，直接編譯
- 某些情況下，編譯器可能能夠處理未優化的模型
- 直接使用 `ants_bees_compatible.onnx` 進行編譯

**選項 B**：使用 onnx-simplifier
```bash
pip install onnx-simplifier
onnxsim ants_bees_compatible.onnx ants_bees_simplified.onnx
```

### 如果步驟 3 失敗（定點分析）

1. 檢查校準圖片是否正確
2. 調整 `radix` 參數（嘗試 7 或 9）
3. 調整 `outlier` 參數（嘗試 0.99 或 0.995）

### 如果編譯失敗

1. 確認 .bie 文件已生成
2. 檢查 `batch_input_params.json` 中的路徑是否正確
3. 查看編譯日誌中的具體錯誤信息

## 快速執行腳本

創建 `deploy.sh` 自動化整個流程：

```bash
#!/bin/bash
# 在 Docker 容器內執行

echo "步驟 1: 優化 ONNX..."
python complete_optimization.py

echo "步驟 2: 定點分析..."
python run_fp_analysis.py

echo "步驟 3: 編譯..."
python /workspace/scripts/batchCompile_520.py -c batch_input_params.json

echo "完成！"
```

## 預期輸出文件

完成後應該有：
- `ants_bees_opt.onnx` - 優化後的 ONNX 模型
- `ants_bees_opt.quan.wqbi.bie` - 定點分析後的模型
- `models_520.nef` - 最終的 NPU 執行文件

## 下一步

完成編譯後，參考 Part-08 PDF 文檔進行 AI Dongle 部署。

