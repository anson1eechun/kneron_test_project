#!/usr/bin/env python3
"""
根據 Part-05 PDF 文檔的方法，使用 Kneron Toolchain 優化 ONNX 模型
參考：Part_05_Onnx檔案轉換(Convert)_ok8.pdf
"""
import ktc
import onnx
import onnxruntime as rt
import os

# 設定路徑
path = '/docker_mount/'
# 使用合併後的模型（單一文件，無外部數據）
file_name = 'ants_bees_merged'
path_onnx = path + file_name + '.onnx'

# 如果合併後的模型不存在，使用原始模型
if not os.path.exists(path_onnx):
    file_name = 'ants_bees'
    path_onnx = path + file_name + '.onnx'

print("=" * 60)
print("Kneron ONNX 模型優化")
print("=" * 60)
print(f"輸入模型: {path_onnx}")

# 檢查文件是否存在
if not os.path.exists(path_onnx):
    print(f"錯誤：找不到文件 {path_onnx}")
    exit(1)

# 載入 ONNX 模型
print("\n正在載入 ONNX 模型...")
onnx_model = onnx.load(path_onnx)

# 降級 IR 版本以兼容工具鏈
print(f"原始 IR 版本: {onnx_model.ir_version}")
if onnx_model.ir_version > 6:
    print("降級 IR 版本到 6...")
    onnx_model.ir_version = 6
    # 降級 opset 版本
    if onnx_model.opset_import:
        for opset in onnx_model.opset_import:
            if opset.version > 11:
                print(f"降級 opset 版本: {opset.version} -> 11")
                opset.version = 11

# 使用 Kneron Toolchain 進行優化
print("正在執行優化...")
try:
    onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(
        onnx_model,
        eliminate_tail=False  # 某些版本不支持 eliminate_tail
    )
except Exception as e:
    print(f"警告: {e}")
    # 嘗試不使用 eliminate_tail
    onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(onnx_model)

# 保存優化後的模型
fna = file_name + '_opt.onnx'
path_opt = path + fna
onnx.save(onnx_opt, path_opt)
print(f'\n✓ 優化完成！已保存到: {path_opt}')

# 驗證優化後的模型
print("\n正在驗證優化後的模型...")
try:
    session = rt.InferenceSession(path_opt)
    model_meta = session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    print(f'✓ 模型驗證通過')
    print(f'  輸入名稱: {input_name}')
    print(f'  模型描述: {model_meta.description}')
    print(f'  模型版本: {model_meta.version}')
except Exception as e:
    print(f'⚠ 模型驗證警告: {e}')

print("\n" + "=" * 60)
print("優化完成！")
print("=" * 60)

