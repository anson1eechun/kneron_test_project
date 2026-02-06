#!/usr/bin/env python3
"""
直接編譯 ONNX 模型為 .nef 文件（跳過定點分析）
使用 ktc.compile API
"""
import ktc
import os
import sys

# 設定路徑
model_path = "/docker_mount/ants_bees_opt_fixed.onnx"
# 如果固定版本不存在，使用原始版本
if not os.path.exists(model_path):
    model_path = "/docker_mount/ants_bees_opt.onnx"
output_path = "/docker_mount/models_520.nef"

print("=" * 60)
print("直接編譯 ONNX 模型為 .nef（跳過定點分析）")
print("=" * 60)
print(f"輸入模型: {model_path}")
print(f"輸出文件: {output_path}")
print()

# 檢查文件是否存在
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型文件 {model_path}")
    sys.exit(1)

try:
    print("[步驟 1] 創建模型配置...")
    # 使用 ktc.ModelConfig 創建配置
    # 參數：id, version, platform, onnx_path
    config = ktc.ModelConfig(
        id=100,
        version="0000",
        platform="520",
        onnx_path=model_path
    )
    print("  [OK] 配置已創建")
    
    print("\n[步驟 2] 執行分析（如果需要）...")
    # 根據文檔，ONNX 模型可能需要先運行 analysis()
    try:
        config.analysis()
        print("  [OK] 分析完成")
    except Exception as e:
        print(f"  [WARNING] 分析步驟失敗: {e}")
        print("  嘗試直接編譯...")
    
    print("\n[步驟 3] 執行編譯...")
    print("  使用 ktc.compile()...")
    
    # ktc.compile 需要一個 ModelConfig 列表
    model_list = [config]
    nef_path = ktc.compile(
        model_list=model_list,
        output_dir="/docker_mount",
        dedicated_output_buffer=True,
        weight_compress=False
    )
    
    print(f"\n[步驟 4] 檢查輸出文件...")
    if os.path.exists(nef_path):
        print(f"  [OK] .nef 文件已生成: {nef_path}")
        file_size = os.path.getsize(nef_path) / (1024 * 1024)
        print(f"  文件大小: {file_size:.2f} MB")
        
        # 如果輸出路徑不同，複製到指定位置
        if nef_path != output_path:
            import shutil
            shutil.copy2(nef_path, output_path)
            print(f"  [OK] 已複製到: {output_path}")
    else:
        print(f"  [WARNING] 文件未找到: {nef_path}")
        # 檢查輸出目錄
        output_dir = "/docker_mount"
        nef_files = [f for f in os.listdir(output_dir) if f.endswith('.nef')]
        if nef_files:
            print(f"  找到 .nef 文件: {nef_files}")
    
    print("\n" + "=" * 60)
    print("[OK] 編譯完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] 編譯失敗: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n嘗試其他方法...")
    print("可能需要先進行定點分析，或使用不同的編譯方法")
    sys.exit(1)

