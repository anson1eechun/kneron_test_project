"""
比較三個模型的推論結果
分析優化過程對模型精度的影響
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

# 三個模型文件
MODELS = {
    '原始模型': 'ants_bees.onnx',
    '優化模型': 'ants_bees_opt.onnx',
    '修復模型': 'ants_bees_opt_fixed.onnx'
}

# 測試圖片
TEST_IMAGES = [
    ('螞蟻', 'data/val/ants/10308379_1b6c72e180.jpg'),
    ('蜜蜂', 'data/val/bees/2525379273_dcb26a516d.jpg'),
]

CLASSES = ['Ant (螞蟻)', 'Bee (蜜蜂)']

def preprocess(image_path):
    """預處理圖片"""
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

def run_inference(model_path, image_path):
    """執行推論"""
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        input_data = preprocess(image_path)
        output = session.run(None, {input_name: input_data})
        raw_result = output[0][0]
        
        # 計算 Softmax
        exp_result = np.exp(raw_result - np.max(raw_result))
        probabilities = exp_result / np.sum(exp_result)
        
        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return raw_result, probabilities, predicted_class
    except Exception as e:
        return None, None, str(e)

print("=" * 80)
print("模型比較測試")
print("=" * 80)

# 檢查模型文件
print("\n📦 檢查模型文件:")
for name, path in MODELS.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) / (1024*1024) if exists else 0
    print(f"  {name:10s}: {path:30s} {'✓' if exists else '✗'} ({size:.1f} MB)")

# 測試每個模型
results = {}
for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        continue
    
    print(f"\n{'='*80}")
    print(f"測試模型: {model_name} ({model_path})")
    print(f"{'='*80}")
    
    results[model_name] = {}
    
    for label, image_path in TEST_IMAGES:
        if not os.path.exists(image_path):
            print(f"  ⚠️  跳過：找不到圖片 {image_path}")
            continue
        
        raw_result, probabilities, predicted = run_inference(model_path, image_path)
        
        if raw_result is None:
            print(f"  ❌ 推論失敗: {predicted}")
            continue
        
        # 計算置信度差異
        sorted_probs = sorted(probabilities, reverse=True)
        confidence_diff = (sorted_probs[0] - sorted_probs[1]) * 100 if len(sorted_probs) > 1 else 0
        
        # 判斷是否正確
        is_correct = (label == '螞蟻' and predicted == CLASSES[0]) or \
                     (label == '蜜蜂' and predicted == CLASSES[1])
        
        status = "✓" if is_correct else "✗"
        
        print(f"\n  {status} 測試圖片: {label} ({os.path.basename(image_path)})")
        print(f"     原始輸出: {raw_result}")
        print(f"     概率分佈:")
        for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            marker = " ←" if i == np.argmax(probabilities) else ""
            print(f"       {cls}: {prob*100:.2f}%{marker}")
        print(f"     預測結果: {predicted}")
        print(f"     置信度差異: {confidence_diff:.2f}%")
        
        results[model_name][label] = {
            'raw': raw_result,
            'probabilities': probabilities,
            'predicted': predicted,
            'correct': is_correct,
            'confidence_diff': confidence_diff
        }

# 總結
print(f"\n{'='*80}")
print("測試總結")
print(f"{'='*80}")

print("\n準確度統計:")
for model_name in results.keys():
    correct_count = sum(1 for r in results[model_name].values() if r['correct'])
    total_count = len(results[model_name])
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"  {model_name:10s}: {correct_count}/{total_count} ({accuracy:.1f}%)")

print("\n平均置信度差異:")
for model_name in results.keys():
    avg_diff = np.mean([r['confidence_diff'] for r in results[model_name].values()])
    print(f"  {model_name:10s}: {avg_diff:.2f}%")

print("\n⚠️  問題分析:")
print("  1. 如果優化模型和修復模型的準確度下降，說明優化過程可能破壞了模型")
print("  2. 如果置信度差異很小（<10%），說明模型不確定，可能是優化導致的精度損失")
print("  3. 建議：使用原始模型 ants_bees.onnx 進行部署，或重新進行優化")

print("\n" + "="*80)

