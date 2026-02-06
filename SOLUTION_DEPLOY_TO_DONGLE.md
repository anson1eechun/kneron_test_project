# 部署到 AI Dongle 的完整解決方案

## 問題分析

根據當前狀態，主要問題是：
1. **Part-05**: ONNX 優化失敗（版本兼容性）
2. **Part-07**: 編譯失敗（需要先完成優化）

## 解決方案：三步驟完成部署

### 步驟 1：重新導出兼容的 ONNX 模型

問題根源：PyTorch 2.10.0 導出的 ONNX 使用 IR 版本 10，但工具鏈只支持 ≤ 6。

**解決方法**：使用 `onnx-simplifier` 進行版本轉換和優化。

### 步驟 2：使用 Kneron Toolchain 優化

根據 PDF 文檔，使用 `ktc.onnx_optimizer.onnx2onnx_flow()` 進行優化。

### 步驟 3：編譯生成 .nef 文件

完成優化後，進行定點分析和編譯，生成 .nef 文件。

## 實施計劃

