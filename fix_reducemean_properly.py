"""
正確替換 ONNX 模型中的 ReduceMean 操作
檢查 ReduceMean 的輸入輸出和屬性，正確替換為 GlobalAveragePool
"""
import onnx
from onnx import helper
import sys

def fix_reducemean(model_path, output_path):
    """替換 ReduceMean 為 GlobalAveragePool"""
    print(f"正在載入模型: {model_path}")
    model = onnx.load(model_path)
    
    print("正在查找 ReduceMean 節點...")
    reducemean_nodes = [node for node in model.graph.node if node.op_type == "ReduceMean"]
    
    if not reducemean_nodes:
        print("未找到 ReduceMean 節點")
        onnx.save(model, output_path)
        return
    
    print(f"找到 {len(reducemean_nodes)} 個 ReduceMean 節點")
    
    # 檢查每個 ReduceMean 節點
    for i, node in enumerate(reducemean_nodes):
        print(f"\n節點 {i+1}: {node.name}")
        print(f"  輸入: {node.input}")
        print(f"  輸出: {node.output}")
        
        # 檢查屬性
        axes = None
        keepdims = 1
        
        for attr in node.attribute:
            print(f"  屬性 {attr.name}: {list(attr.ints) if attr.ints else attr.i}")
            if attr.name == "axes":
                axes = list(attr.ints) if attr.ints else None
            elif attr.name == "keepdims":
                keepdims = attr.i
        
        # 檢查輸入的形狀（通過 value_info）
        input_shape = None
        for vi in model.graph.value_info:
            if vi.name == node.input[0]:
                if vi.type.tensor_type.shape.dim:
                    input_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in vi.type.tensor_type.shape.dim]
                break
        
        print(f"  輸入形狀: {input_shape}")
        
        # 如果沒有 axes 屬性，檢查是否有常量輸入
        if axes is None and len(node.input) > 1:
            # 檢查第二個輸入是否是常量
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    axes = list(init.int32_data) if init.int32_data else list(init.int64_data)
                    print(f"  從常量找到 axes: {axes}")
                    break
        
        # 判斷是否為 Global Average Pooling
        # 通常：輸入是 [N, C, H, W]，axes=[2, 3]，keepdims=1
        is_gap = False
        if input_shape and len(input_shape) == 4:
            if axes == [2, 3] or axes == [-2, -1]:
                is_gap = True
            elif axes is None and input_shape[2] > 1 and input_shape[3] > 1:
                # 可能是全局池化
                is_gap = True
        
        if is_gap:
            print(f"  [OK] 檢測到 Global Average Pooling，準備替換...")
            
            # 創建 GlobalAveragePool 節點
            gap_node = helper.make_node(
                "GlobalAveragePool",
                inputs=[node.input[0]],
                outputs=node.output,
                name=node.name.replace("ReduceMean", "GAP") if "ReduceMean" in node.name else node.name + "_gap"
            )
            
            # 替換節點
            node_index = list(model.graph.node).index(node)
            model.graph.node.remove(node)
            model.graph.node.insert(node_index, gap_node)
            print(f"  [OK] 已替換為 GlobalAveragePool")
        else:
            print(f"  [WARNING] 無法確定是否為 GAP (axes={axes})，跳過")
    
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
    
    fix_reducemean(input_path, output_path)
    print("\n完成！")

