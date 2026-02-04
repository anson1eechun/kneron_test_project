# Part-07 模型編譯說明

## 已完成的準備工作

✅ **步驟一：準備校準圖片**
- 已建立 `imgs/` 資料夾
- 已複製 30 張校準圖片（螞蟻和蜜蜂各 15 張）

✅ **步驟二：建立 input_params.json**
- 已建立符合 Kneron 工具鏈格式的配置文件
- 包含模型資訊、輸入配置和預處理參數

✅ **步驟三：建立 batch_input_params.json**
- 已建立批次編譯配置文件
- 模型 ID 設為 100

## 遇到的問題

### 問題 1：batchCompile_520.py 只支持 .bie 文件

`batchCompile_520.py` 腳本只支持已經進行過定點分析（Fix Point Analysis）的模型文件（.bie 格式），不支持直接編譯 ONNX 文件。

**錯誤訊息：**
```
ValueError: Currently, batch compile only support models after fix point analysis.
```

### 問題 2：需要先進行定點分析

在編譯之前，需要先使用 `knerex`（FP Analyzer）將 ONNX 模型轉換為 .bie 格式。

## 解決方案

### 方案 1：使用完整的編譯流程（推薦）

根據 Kneron 工具鏈的標準流程，需要兩個步驟：

1. **定點分析（Fix Point Analysis）**：將 ONNX 轉換為 .bie
2. **編譯（Compilation）**：將 .bie 編譯為 .nef

### 方案 2：查找整合腳本

用戶提到的 `fpAnalyserBatchCompile_520.py` 可能是一個整合了定點分析和編譯的腳本，但在當前工具鏈版本中未找到。

## 當前配置文件

### input_params.json
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
        "img_preprocess_method": "normalize",
        "img_channel": "RGB",
        "radix": 8,
        "keep_aspect_ratio": false,
        "pad_mode": 0,
        "p_crop": {
            "crop_x": 0,
            "crop_y": 0,
            "crop_w": 0,
            "crop_h": 0
        },
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}
```

### batch_input_params.json
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

## 下一步建議

1. **查找正確的編譯腳本**：確認工具鏈中是否有整合定點分析和編譯的腳本
2. **分步執行**：先執行定點分析生成 .bie 文件，再執行編譯
3. **聯繫 Kneron 支持**：獲取最新的編譯流程文檔

## 相關文件位置

- 校準圖片：`imgs/`（30 張圖片）
- 輸入配置：`input_params.json`
- 批次配置：`batch_input_params.json`
- ONNX 模型：`ants_bees_opt.onnx`

