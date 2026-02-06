"""
重新導出兼容的 ONNX 模型
使用較低的 opset 版本並確保兼容性
"""
import torch
import torch.nn as nn
from torchvision import models
import onnx
import os

# 載入訓練好的模型權重（如果有的話）
# 這裡我們重新創建模型結構並導出

print("=" * 60)
print("重新導出兼容的 ONNX 模型")
print("=" * 60)

# 創建模型結構
print("正在創建 ResNet50 模型結構...")
try:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
except AttributeError:
    model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 2)

# 載入權重（如果之前有保存）
weight_files = ['ants_bees_model.pth', 'model.pth', 'best_model.pth']
weight_loaded = False

for weight_file in weight_files:
    try:
        if os.path.exists(weight_file):
            checkpoint = torch.load(weight_file, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"[OK] 已載入訓練好的權重: {weight_file}")
            weight_loaded = True
            break
    except Exception as e:
        continue

if not weight_loaded:
    print("[WARNING] 未找到權重文件，將使用 ImageNet 預訓練權重")
    print("  注意：這將使用預訓練權重而非訓練好的螞蟻/蜜蜂分類權重")
    print("  建議：先運行 train_resnet50.py 訓練模型")

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

print(f"[OK] ONNX 模型已導出: {output_file}")

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
    print("  [OK] 模型驗證通過")
except Exception as e:
    print(f"  ⚠ 模型檢查警告: {e}")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

