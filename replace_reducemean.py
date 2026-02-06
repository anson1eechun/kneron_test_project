"""
替換 ONNX 模型中的 ReduceMean 操作
ResNet50 的 Global Average Pooling 使用 ReduceMean，但 Kneron 編譯器不支持
我們需要將其替換為 GlobalAveragePool 或其他支持的操作
"""
import onnx
from onnx import helper, TensorProto
import numpy as np

def replace_reducemean_with_gap(model_path, output_path):
    """將 ReduceMean 替換為 GlobalAveragePool"""
    print(f"正在載入模型: {model_path}")
    model = onnx.load(model_path)
    
    print("正在查找 ReduceMean 節點...")
    reducemean_nodes = [node for node in model.graph.node if node.op_type == "ReduceMean"]
    
    if not reducemean_nodes:
        print("未找到 ReduceMean 節點，模型可能已經優化過")
        onnx.save(model, output_path)
        return
    
    print(f"找到 {len(reducemean_nodes)} 個 ReduceMean 節點")
    
    # 替換每個 ReduceMean 節點
    for node in reducemean_nodes:
        print(f"  替換節點: {node.name}")
        
        # 檢查 ReduceMean 的屬性
        # Global Average Pooling 通常使用 axes=[2, 3] 和 keepdims=1
        axes = None
        keepdims = 1
        
        for attr in node.attribute:
            if attr.name == "axes":
                axes = list(attr.ints)
            elif attr.name == "keepdims":
                keepdims = attr.i
        
        # 如果 axes 是 [2, 3] 或類似，這是 Global Average Pooling
        if axes and len(axes) == 2 and axes == [2, 3]:
            print(f"    檢測到 Global Average Pooling (axes={axes})")
            
            # 創建 GlobalAveragePool 節點
            gap_node = helper.make_node(
                "GlobalAveragePool",
                inputs=[node.input[0]],
                outputs=node.output,
                name=node.name + "_gap"
            )
            
            # 替換節點
            node_index = list(model.graph.node).index(node)
            model.graph.node.remove(node)
            model.graph.node.insert(node_index, gap_node)
            print(f"    [OK] 已替換為 GlobalAveragePool")
        else:
            print(f"    [WARNING] 無法替換 (axes={axes})，保留原節點")
    
    print(f"\n正在保存修改後的模型: {output_path}")
    onnx.save(model, output_path)
    
    # 驗證模型
    try:
        onnx.checker.check_model(model)
        print("[OK] 模型驗證通過")
    except Exception as e:
        print(f"[WARNING] 模型驗證警告: {e}")

if __name__ == "__main__":
    input_path = "ants_bees_opt.onnx"
    output_path = "ants_bees_opt_fixed.onnx"
    
    replace_reducemean_with_gap(input_path, output_path)
    print("\n完成！")

