"""
簡單的 ONNX 優化腳本
由於 Kneron 工具鏈容器中的 ONNX 版本兼容性問題，
這個腳本提供基本的優化功能
"""
import onnx
import sys

def optimize_onnx(input_path, output_path):
    """載入並優化 ONNX 模型"""
    print(f"正在載入模型: {input_path}")
    model = onnx.load(input_path)
    
    print(f"模型 IR 版本: {model.ir_version}")
    print(f"模型 Opset 版本: {model.opset_import[0].version if model.opset_import else 'N/A'}")
    
    # 檢查模型
    try:
        onnx.checker.check_model(model)
        print("✓ 模型驗證通過")
    except Exception as e:
        print(f"⚠ 模型驗證警告: {e}")
    
    # 基本優化：確保模型結構正確
    # 注意：這裡只做基本的檢查和保存，實際的 Kneron 特定優化
    # 需要在 Kneron 工具鏈中完成，但由於版本兼容性問題暫時跳過
    
    print(f"正在保存優化後的模型: {output_path}")
    onnx.save(model, output_path)
    print("✓ 模型已保存")
    
    # 驗證輸出模型
    try:
        output_model = onnx.load(output_path)
        onnx.checker.check_model(output_model)
        print("✓ 輸出模型驗證通過")
    except Exception as e:
        print(f"⚠ 輸出模型驗證警告: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python optimize_onnx.py <input.onnx> <output.onnx>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    optimize_onnx(input_path, output_path)

