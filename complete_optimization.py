"""
完整的 ONNX 優化流程
根據 Part-05 PDF 文檔的方法
"""
import ktc
import onnx
import onnxruntime as rt
import os
import sys

# 設定路徑
path = '/docker_mount/'
file_name = 'ants_bees_compatible'  # 使用兼容版本
path_onnx = path + file_name + '.onnx'

# 如果兼容版本不存在，嘗試其他版本
if not os.path.exists(path_onnx):
    file_name = 'ants_bees_merged'
    path_onnx = path + file_name + '.onnx'
    
if not os.path.exists(path_onnx):
    file_name = 'ants_bees'
    path_onnx = path + file_name + '.onnx'

print("=" * 60)
print("Kneron ONNX 模型優化（完整流程）")
print("=" * 60)
print(f"輸入模型: {path_onnx}")

# 檢查文件是否存在
if not os.path.exists(path_onnx):
    print(f"❌ 錯誤：找不到文件 {path_onnx}")
    print("\n請先執行 fix_onnx_export.py 重新導出模型")
    sys.exit(1)

# 載入 ONNX 模型
print("\n[步驟 1] 載入 ONNX 模型...")
try:
    onnx_model = onnx.load(path_onnx)
    print(f"  ✓ 模型載入成功")
    print(f"  IR 版本: {onnx_model.ir_version}")
    if onnx_model.opset_import:
        for opset in onnx_model.opset_import:
            print(f"  Opset 版本: {opset.version}")
except Exception as e:
    print(f"  ❌ 載入失敗: {e}")
    sys.exit(1)

# 檢查並調整版本
print("\n[步驟 2] 檢查版本兼容性...")
if onnx_model.ir_version > 6:
    print(f"  ⚠ IR 版本 {onnx_model.ir_version} 可能不兼容，嘗試調整...")
    # 注意：直接修改可能導致問題，但先嘗試
    original_ir = onnx_model.ir_version
    onnx_model.ir_version = 6
    print(f"  IR 版本: {original_ir} -> 6")

# 使用 Kneron Toolchain 進行優化
print("\n[步驟 3] 執行 Kneron 優化...")
try:
    print("  使用 ktc.onnx_optimizer.onnx2onnx_flow()...")
    onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(
        onnx_model,
        eliminate_tail=True  # 根據 PDF 文檔使用 eliminate_tail
    )
    print("  ✓ 優化成功")
except Exception as e1:
    print(f"  ⚠ 第一次嘗試失敗: {e1}")
    print("  嘗試不使用 eliminate_tail...")
    try:
        onnx_opt = ktc.onnx_optimizer.onnx2onnx_flow(onnx_model)
        print("  ✓ 優化成功（不使用 eliminate_tail）")
    except Exception as e2:
        print(f"  ❌ 優化失敗: {e2}")
        print("\n建議：")
        print("1. 使用 fix_onnx_export.py 重新導出模型")
        print("2. 或聯繫 Kneron 支持獲取更新的工具鏈")
        sys.exit(1)

# 保存優化後的模型
fna = 'ants_bees_opt.onnx'
path_opt = path + fna
print(f"\n[步驟 4] 保存優化後的模型...")
try:
    onnx.save(onnx_opt, path_opt)
    print(f"  ✓ 已保存到: {path_opt}")
except Exception as e:
    print(f"  ❌ 保存失敗: {e}")
    sys.exit(1)

# 驗證優化後的模型
print("\n[步驟 5] 驗證優化後的模型...")
try:
    session = rt.InferenceSession(path_opt)
    model_meta = session.get_modelmeta()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"  ✓ 模型驗證通過")
    print(f"  輸入名稱: {input_name}")
    print(f"  輸出名稱: {output_name}")
    print(f"  模型描述: {model_meta.description}")
except Exception as e:
    print(f"  ⚠ 驗證警告: {e}")

print("\n" + "=" * 60)
print("✓ ONNX 優化完成！")
print(f"  優化後的模型: {path_opt}")
print("=" * 60)
print("\n下一步：執行 Part-07 編譯流程")

