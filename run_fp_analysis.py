#!/usr/bin/env python3
"""
執行定點分析 (FP Analysis) 將 ONNX 轉換為 .bie 格式
"""
import sys
import os

# 添加工具鏈路徑
sys.path.insert(0, '/workspace/scripts')

from utils.run_knerex import run_knerex
from utils.load_config import ModelConfig

def main():
    # 設定路徑
    input_params_path = "/docker_mount/input_params.json"
    model_path = "/docker_mount/ants_bees_merged.onnx"
    
    print("=" * 60)
    print("開始執行定點分析 (FP Analysis)")
    print("=" * 60)
    print(f"模型文件: {model_path}")
    print(f"配置文件: {input_params_path}")
    print()
    
    try:
        # 載入配置
        print("載入配置...")
        config = ModelConfig(input_params_path, model_path)
        
        # 執行定點分析
        print("執行定點分析...")
        result = run_knerex(config, thread_num=1, hardware=520)
        
        print()
        print("=" * 60)
        print("定點分析完成！")
        print(f"生成的 .bie 文件: {result}")
        print("=" * 60)
        
        # 檢查文件是否存在
        if os.path.exists(result):
            print(f"✓ 文件已生成: {result}")
            # 複製到掛載目錄以便在 Windows 中訪問
            import shutil
            output_name = os.path.basename(result)
            docker_mount_path = f"/docker_mount/{output_name}"
            shutil.copy2(result, docker_mount_path)
            print(f"✓ 已複製到: {docker_mount_path}")
        else:
            print(f"⚠ 警告: 文件未找到: {result}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

