"""
重新導出兼容的 ONNX 模型
使用較低的 opset 版本並確保兼容性
"""
import torch
import torch.nn as nn
from torchvision import models
import onnx

# 載入訓練好的模型權重（如果有的話）
# 這裡我們重新創建模型結構並導出

print("=" * 60)
print("重新導出兼容的 ONNX 模型")
print("=" * 60)

# 創建模型結構
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# 載入權重（如果之前有保存）
try:
    checkpoint = torch.load('ants_bees_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    print("✓ 已載入訓練好的權重")
except:
    print("⚠ 未找到權重文件，將使用預訓練模型結構")
    # 如果沒有權重，至少確保模型結構正確
    pass

model.eval()

# 創建虛擬輸入
dummy_input = torch.randn(1, 3, 224, 224)

# 導出 ONNX（使用兼容的參數）
output_file = "ants_bees_compatible.onnx"

print(f"\n正在導出 ONNX 模型到: {output_file}")
print("使用參數: opset_version=11, do_constant_folding=True")

torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['output'],
    opset_version=11,  # 使用較低的 opset 版本
    do_constant_folding=True,  # 啟用常量折疊
    dynamic_axes=None,  # 不使用動態軸
    export_params=True,
    verbose=False
)

print(f"✓ ONNX 模型已導出: {output_file}")

# 檢查導出的模型
print("\n檢查導出的模型...")
try:
    model_onnx = onnx.load(output_file)
    print(f"  IR 版本: {model_onnx.ir_version}")
    if model_onnx.opset_import:
        for opset in model_onnx.opset_import:
            print(f"  Opset 版本: {opset.version}")
    
    # 驗證模型
    onnx.checker.check_model(model_onnx)
    print("  ✓ 模型驗證通過")
except Exception as e:
    print(f"  ⚠ 模型檢查警告: {e}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

