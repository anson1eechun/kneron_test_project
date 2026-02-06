"""
簡單的推論測試腳本
用於快速驗證模型是否正常工作
"""
import onnxruntime as rt
import numpy as np
from PIL import Image
import os
import sys

# 設置輸出編碼（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 設定模型路徑
MODEL_PATH = "ants_bees_opt_fixed.onnx"
TEST_IMAGE = "data/val/ants/10308379_1b6c72e180.jpg"

# 類別名稱
CLASSES = ['Ant (螞蟻)', 'Bee (蜜蜂)']

print("=" * 60)
print("本地推論測試")
print("=" * 60)

# 檢查模型文件
if not os.path.exists(MODEL_PATH):
    print(f"❌ 錯誤：找不到模型文件 {MODEL_PATH}")
    print("\n可用的模型文件:")
    for f in os.listdir('.'):
        if f.endswith('.onnx') and not f.endswith('.data'):
            print(f"  - {f}")
    exit(1)

# 檢查測試圖片
if not os.path.exists(TEST_IMAGE):
    print(f"❌ 錯誤：找不到測試圖片 {TEST_IMAGE}")
    exit(1)

print(f"\n📦 正在載入模型: {MODEL_PATH}")
try:
    session = rt.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"✅ 模型載入成功")
    print(f"   輸入名稱: {input_name}")
    print(f"   輸入形狀: {session.get_inputs()[0].shape}")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n🖼️  正在讀取圖片: {TEST_IMAGE}")
try:
    # 預處理
    img = Image.open(TEST_IMAGE).convert('RGB')
    img = img.resize((256, 256))
    
    # Center Crop 224x224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # 轉為 numpy array 並標準化
    img_data = np.array(img).astype('float32') / 255.0
    
    # 標準化
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std
    img_data = img_data.astype('float32')
    
    # 調整維度 HWC -> CHW -> BCHW
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    
    print(f"✅ 圖片預處理完成")
    print(f"   圖片形狀: {img_data.shape}")
except Exception as e:
    print(f"❌ 圖片處理失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n🔍 開始推論...")
try:
    # 執行推論
    output = session.run(None, {input_name: img_data})
    raw_result = output[0][0]
    
    # 計算 Softmax
    exp_result = np.exp(raw_result - np.max(raw_result))
    probabilities = exp_result / np.sum(exp_result)
    
    # 找出預測類別
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASSES[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    print(f"\n{'=' * 60}")
    print(f"推論結果")
    print(f"{'=' * 60}")
    print(f"原始輸出: {raw_result}")
    print(f"\n概率分佈:")
    for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
        marker = " ←" if i == predicted_idx else ""
        print(f"  {cls}: {prob*100:.2f}%{marker}")
    print(f"\n預測結果: {predicted_class}")
    print(f"置信度: {confidence*100:.2f}%")
    print(f"{'=' * 60}\n")
    
except Exception as e:
    print(f"❌ 推論失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("✅ 測試完成！")

