# Part-05 ONNX 模型優化說明

## 問題說明

在執行 Kneron 工具鏈的 ONNX 優化步驟時，遇到了版本兼容性問題：

- **問題**：Kneron 工具鏈容器中的 `onnx2onnx.py` 腳本依賴於舊版 ONNX API (`onnx.optimizer`)，但容器中安裝的 ONNX 版本已經移除了這個模組。
- **錯誤訊息**：`ImportError: cannot import name 'optimizer' from 'onnx'`

## 臨時解決方案

由於優化工具鏈的兼容性問題，目前採用了以下方案：

1. **直接使用原始模型**：`ants_bees.onnx` 已經是一個有效的 ONNX 模型
2. **創建優化版本**：已將 `ants_bees.onnx` 複製為 `ants_bees_opt.onnx`，以便後續流程可以繼續

## 注意事項

- 原始模型 (`ants_bees.onnx`) 已經包含了 PyTorch 導出時的一些優化
- 如果後續編譯步驟遇到問題，可能需要：
  1. 聯繫 Kneron 支持獲取更新的工具鏈版本
  2. 或者手動進行特定的優化（如 BN 融合等）
  3. 或者使用 Kneron 提供的其他優化工具

## 下一步

可以繼續進行：
- **Part-06**：軟體模擬推論
- **Part-07**：編譯成硬體格式

如果編譯時遇到與模型結構相關的錯誤，可能需要回到這一步進行真正的優化。

