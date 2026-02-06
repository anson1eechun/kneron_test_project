#!/usr/bin/env python3
"""
本地 ONNX 模型推論工具
支持單張圖片、批量處理和詳細結果顯示
"""
import onnxruntime as rt
import numpy as np
from PIL import Image
import os
import sys
import argparse
import glob
from pathlib import Path

# 設置輸出編碼（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 類別名稱
CLASSES = ['Ant (螞蟻)', 'Bee (蜜蜂)']

def preprocess(image_path):
    """
    預處理圖片：調整大小、中心裁切、標準化
    與訓練時的預處理完全一致
    """
    # 讀取圖片
    img = Image.open(image_path).convert('RGB')
    
    # 預處理：調整大小 -> 中心裁切 -> 標準化
    img = img.resize((256, 256))
    
    # Center Crop 224x224
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # 轉為 numpy array 並標準化
    img_data = np.array(img).astype('float32') / 255.0
    
    # 標準化 (Normalize) mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std
    
    # 確保數據類型是 float32
    img_data = img_data.astype('float32')
    
    # 調整維度 HWC -> CHW (3, 224, 224)
    img_data = img_data.transpose(2, 0, 1)
    
    # 增加 Batch 維度 -> (1, 3, 224, 224)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def run_inference(session, input_name, image_path, show_details=True):
    """
    執行推論
    
    Args:
        session: ONNX Runtime Session
        input_name: 輸入層名稱
        image_path: 圖片路徑
        show_details: 是否顯示詳細信息
    
    Returns:
        predicted_class: 預測類別名稱
        confidence: 置信度 (0-1)
        raw_output: 原始輸出數組
    """
    # 預處理圖片
    input_data = preprocess(image_path)
    
    # 執行推論
    output = session.run(None, {input_name: input_data})
    
    # 解析結果
    raw_result = output[0][0]  # 取得第一張圖的輸出
    
    # 計算 Softmax 得到概率
    exp_result = np.exp(raw_result - np.max(raw_result))  # 數值穩定性
    probabilities = exp_result / np.sum(exp_result)
    
    # 找出預測類別
    predicted_idx = np.argmax(probabilities)
    predicted_class = CLASSES[predicted_idx]
    confidence = probabilities[predicted_idx]
    
    if show_details:
        print(f"\n{'='*60}")
        print(f"圖片: {image_path}")
        print(f"{'='*60}")
        print(f"原始輸出: {raw_result}")
        print(f"概率分佈:")
        for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            marker = " ←" if i == predicted_idx else ""
            print(f"  {cls}: {prob*100:.2f}%{marker}")
        print(f"\n預測結果: {predicted_class}")
        print(f"置信度: {confidence*100:.2f}%")
        print(f"{'='*60}\n")
    
    return predicted_class, confidence, raw_result

def main():
    parser = argparse.ArgumentParser(
        description='使用訓練好的 ResNet50 模型進行螞蟻/蜜蜂分類推論',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單張圖片推論
  python inference_local.py -i data/val/ants/image.jpg
  
  # 批量處理資料夾
  python inference_local.py -i data/val/ants/ --batch
  
  # 使用特定模型
  python inference_local.py -i image.jpg -m ants_bees_opt_fixed.onnx
  
  # 簡潔輸出模式
  python inference_local.py -i image.jpg --quiet
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='輸入圖片路徑或資料夾路徑'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='ants_bees_opt_fixed.onnx',
        help='ONNX 模型文件路徑 (預設: ants_bees_opt_fixed.onnx)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='批量處理模式（當輸入是資料夾時）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='簡潔輸出模式（只顯示預測結果）'
    )
    
    parser.add_argument(
        '--ext',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='批量處理時的文件擴展名 (預設: .jpg .jpeg .png .bmp)'
    )
    
    args = parser.parse_args()
    
    # 檢查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 錯誤：找不到模型文件 {args.model}")
        print(f"\n可用的模型文件:")
        onnx_files = glob.glob("*.onnx")
        for f in onnx_files:
            if not f.endswith('.data'):
                print(f"  - {f}")
        sys.exit(1)
    
    # 載入模型
    print(f"📦 正在載入模型: {args.model}")
    try:
        session = rt.InferenceSession(args.model)
        input_name = session.get_inputs()[0].name
        print(f"✅ 模型載入成功")
        print(f"   輸入名稱: {input_name}")
        print(f"   輸入形狀: {session.get_inputs()[0].shape}")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        sys.exit(1)
    
    # 處理輸入
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 單張圖片
        if not input_path.exists():
            print(f"❌ 錯誤：找不到圖片文件 {args.input}")
            sys.exit(1)
        
        print(f"\n🔍 開始推論...")
        predicted_class, confidence, _ = run_inference(
            session, input_name, str(input_path),
            show_details=not args.quiet
        )
        
        if args.quiet:
            print(f"{predicted_class} ({confidence*100:.1f}%)")
    
    elif input_path.is_dir():
        # 批量處理
        if not args.batch:
            print(f"❌ 錯誤：輸入是資料夾，請使用 --batch 參數啟用批量處理")
            sys.exit(1)
        
        # 收集所有圖片文件
        image_files = []
        for ext in args.ext:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ 錯誤：在 {args.input} 中找不到圖片文件")
            sys.exit(1)
        
        print(f"\n📁 找到 {len(image_files)} 張圖片")
        print(f"🔍 開始批量推論...\n")
        
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 處理: {img_path.name}")
            try:
                predicted_class, confidence, _ = run_inference(
                    session, input_name, str(img_path),
                    show_details=not args.quiet
                )
                results.append({
                    'file': img_path.name,
                    'class': predicted_class,
                    'confidence': confidence
                })
                
                if args.quiet:
                    print(f"  → {predicted_class} ({confidence*100:.1f}%)")
            except Exception as e:
                print(f"  ❌ 處理失敗: {e}")
                results.append({
                    'file': img_path.name,
                    'class': 'ERROR',
                    'confidence': 0.0
                })
        
        # 統計結果
        print(f"\n{'='*60}")
        print(f"📊 批量推論統計")
        print(f"{'='*60}")
        ant_count = sum(1 for r in results if r['class'] == CLASSES[0])
        bee_count = sum(1 for r in results if r['class'] == CLASSES[1])
        error_count = sum(1 for r in results if r['class'] == 'ERROR')
        
        print(f"總計: {len(results)} 張圖片")
        print(f"  {CLASSES[0]}: {ant_count} 張 ({ant_count/len(results)*100:.1f}%)")
        print(f"  {CLASSES[1]}: {bee_count} 張 ({bee_count/len(results)*100:.1f}%)")
        if error_count > 0:
            print(f"  錯誤: {error_count} 張")
        print(f"{'='*60}\n")
    
    else:
        print(f"❌ 錯誤：無效的輸入路徑 {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()

