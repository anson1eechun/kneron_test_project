"""
降級 ONNX 模型的 IR 版本以兼容舊版 onnxruntime
"""
import onnx

def downgrade_onnx_ir(input_path, output_path, target_ir_version=6):
    """降級 ONNX 模型的 IR 版本"""
    print(f"正在載入模型: {input_path}")
    model = onnx.load(input_path)
    
    print(f"原始 IR 版本: {model.ir_version}")
    print(f"目標 IR 版本: {target_ir_version}")
    
    # 降級 IR 版本
    model.ir_version = target_ir_version
    
    # 降級 opset 版本（如果需要）
    if model.opset_import:
        for opset in model.opset_import:
            if opset.version > 11:
                print(f"降級 opset 版本: {opset.version} -> 11")
                opset.version = 11
    
    print(f"正在保存降級後的模型: {output_path}")
    onnx.save(model, output_path)
    
    # 驗證
    try:
        onnx.checker.check_model(model)
        print("✓ 模型驗證通過")
    except Exception as e:
        print(f"⚠ 模型驗證警告: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("使用方法: python downgrade_onnx_ir.py <input.onnx> <output.onnx>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    downgrade_onnx_ir(input_path, output_path)

