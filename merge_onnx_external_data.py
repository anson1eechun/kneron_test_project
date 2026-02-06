"""
合併 ONNX 模型的外部數據文件為單一文件
"""
import onnx
import sys

def merge_external_data(input_path, output_path):
    """將外部數據合併到 ONNX 模型中"""
    print(f"正在載入模型: {input_path}")
    model = onnx.load(input_path)
    
    print("正在合併外部數據...")
    # 使用 load_external_data_for_model 來加載外部數據
    # 然後保存為單一文件
    onnx.save_model(model, output_path, save_as_external_data=False)
    
    print(f"✓ 已保存為單一文件: {output_path}")

if __name__ == "__main__":
    input_path = "ants_bees.onnx"
    output_path = "ants_bees_merged.onnx"
    
    merge_external_data(input_path, output_path)
    print("完成！")

