# Kneron ResNet50 螞蟻/蜜蜂分類專案

這是一個使用 ResNet50 進行螞蟻和蜜蜂圖像分類的專案，專為 Kneron NPU 部署設計。

## 專案概述

- **模型架構**: ResNet50
- **任務**: 二分類（螞蟻 vs 蜜蜂）
- **訓練準確度**: 96.1%
- **驗證準確度**: 96.1%

## 專案結構

```
kneron_project/
├── data/                    # 訓練和驗證數據
│   ├── train/              # 訓練集（244 張圖片）
│   │   ├── ants/          # 螞蟻圖片（124 張）
│   │   └── bees/          # 蜜蜂圖片（121 張）
│   └── val/                # 驗證集（153 張圖片）
│       ├── ants/          # 螞蟻圖片（70 張）
│       └── bees/          # 蜜蜂圖片（83 張）
├── train_resnet50.py       # 訓練腳本
├── inference_test.py       # Docker 推論測試腳本
├── inference_test_local.py # 本地推論測試腳本
├── ants_bees.onnx         # 原始 ONNX 模型
├── ants_bees_opt.onnx     # 優化後的 ONNX 模型
└── requirements.txt        # Python 依賴套件

```

## 快速開始

### 1. 安裝依賴

```bash
py -m pip install -r requirements.txt
```

### 2. 訓練模型

```bash
py train_resnet50.py
```

訓練完成後會自動生成 `ants_bees.onnx` 模型文件。

### 3. 測試推論

**本地測試：**
```bash
py inference_test_local.py
```

**Docker 容器測試：**
```bash
docker run --rm -v ${PWD}:/docker_mount kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && pip install --upgrade onnxruntime && python /docker_mount/inference_test.py"
```

## 專案進度

- ✅ **Part-04**: 模型訓練與 ONNX 匯出
- ✅ **Part-05**: ONNX 模型優化（遇到版本兼容性問題，已記錄）
- ✅ **Part-06**: 軟體模擬推論測試
- ⏳ **Part-07**: 編譯成硬體格式（待進行）

## 測試結果

### 推論測試結果

**Test 1 - 螞蟻圖片：**
- Raw Output: [ 1.5317764 -1.6757964]
- 預測結果: **Ant (螞蟻)** ✓

**Test 2 - 蜜蜂圖片：**
- Raw Output: [-1.3074203  1.2468289]
- 預測結果: **Bee (蜜蜂)** ✓

## 注意事項

1. **大文件警告**: 某些 ONNX 模型文件超過 50MB，GitHub 建議使用 Git LFS
2. **ONNX 版本**: 模型使用 ONNX IR 版本 10，某些舊版工具可能不兼容
3. **Docker 測試**: 需要在容器中升級 `onnxruntime` 才能正常運行

## 相關文件

- `PART05_OPTIMIZATION_NOTE.md` - ONNX 優化步驟說明
- `PART06_INFERENCE_NOTE.md` - 推論測試說明

## 技術棧

- **框架**: PyTorch
- **模型**: ResNet50 (ImageNet 預訓練)
- **格式**: ONNX
- **目標平台**: Kneron NPU

## 授權

本專案僅供學習和研究使用。

