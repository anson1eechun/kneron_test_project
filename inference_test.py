import onnxruntime as rt
import numpy as np
from PIL import Image
import os
import sys

# 設定模型路徑 (在 Docker 內的路徑)
MODEL_PATH = "/docker_mount/ants_bees_opt.onnx"

# 類別名稱
CLASSES = ['Ant (螞蟻)', 'Bee (蜜蜂)']

def preprocess(image_path):
    # 讀取圖片
    img = Image.open(image_path).convert('RGB')
    
    # 預處理：調整大小 -> 中心裁切 -> 標準化
    # 這些步驟必須跟訓練時(train_resnet50.py)完全一致
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

def run_inference(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    # 建立 ONNX Runtime Session
    session = rt.InferenceSession(MODEL_PATH)
    
    # 取得輸入層名稱
    input_name = session.get_inputs()[0].name
    
    # 預處理圖片
    input_data = preprocess(image_path)
    
    # 執行推論
    output = session.run(None, {input_name: input_data})
    
    # 解析結果
    raw_result = output[0][0] # 取得第一張圖的輸出
    predicted_idx = np.argmax(raw_result) # 找出數值最大的索引
    
    print(f"-----------------------------")
    print(f"測試圖片: {image_path}")
    print(f"Raw Output: {raw_result}")
    print(f"預測結果: {CLASSES[predicted_idx]}")
    print(f"-----------------------------")

if __name__ == "__main__":
    # 您可以隨意指定一張圖片的路徑 (Docker 內的路徑)
    # 這裡我們預設抓 data/val 資料夾裡面的圖片來測
    
    # 測試一張螞蟻
    print("\n[Test 1] Testing Ant...")
    # 注意：請確認您的資料夾結構，若無此檔請自行更換路徑
    test_img_1 = "/docker_mount/data/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg" 
    # 如果找不到特定檔名，這行會抓資料夾第一張
    if not os.path.exists(test_img_1):
        import glob
        files = glob.glob("/docker_mount/data/val/ants/*.jpg")
        if files: test_img_1 = files[0]
            
    run_inference(test_img_1)

    # 測試一張蜜蜂
    print("\n[Test 2] Testing Bee...")
    test_img_2 = "/docker_mount/data/val/bees/21399619_3e61e5bb6f.jpg"
    if not os.path.exists(test_img_2):
        import glob
        files = glob.glob("/docker_mount/data/val/bees/*.jpg")
        if files: test_img_2 = files[0]

    run_inference(test_img_2)

