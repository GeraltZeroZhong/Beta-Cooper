import os
import sys
import argparse
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 确保能引用本地包
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 引用重构后的包
from beta_cooper.validator import BarrelValidator
from beta_cooper.geometry import BarrelGeometry

# --- 核心处理逻辑 (Worker) ---
def analyze_structure(pdb_path, return_full_data=False):
    """
    通用分析函数。
    :param return_full_data: 如果为 True，返回包含调试数据的结果 (用于 Single Mode)
    :return: 结果字典
    """
    pdb_path = os.path.abspath(pdb_path)
    filename = os.path.basename(pdb_path)
    stem = Path(pdb_path).stem
    
    start_time = time.time()
    
    # 默认结果模板
    result = {
        "filename": filename, "id": stem,
        "status": "UNKNOWN", "confidence": 0.0,
        "n_strands": np.nan, "shear_S": np.nan, 
        "radius": np.nan, "tilt": np.nan, "height": np.nan,
        "processing_time": 0.0, "issue": "None"
    }
    
    try:
        # 1. Validator
        validator = BarrelValidator(pdb_path)
        v_res = validator.validate()
        
        result.update({
            "status": v_res['status'],
            "confidence": v_res['confidence'],
            "issue": v_res['issue'],
            **v_res['metrics']
        })
        
        # 2. Geometry (仅当有提取出的片段时运行)
        beta_segments = v_res.get('debug_segments')
        all_coords = v_res.get('debug_coords')
        
        if beta_segments and len(beta_segments) > 0:
            geo = BarrelGeometry(segments=beta_segments, all_coords=all_coords)
            params = geo.get_summary()
            
            result.update({
                "n_strands": params['n_strands'],
                "shear_S": params['shear_S'],
                "radius": params['radius'],
                "tilt": params['tilt_angle'],
                "height": params['height']
            })
            
            # 单文件模式需要详细的几何对象来打印更多信息
            if return_full_data:
                result['debug_geo'] = geo

        elif not v_res['is_valid']:
             # 如果无效且没有片段，通常是提取失败
             pass
        else:
             result["status"] = "FAIL_NO_BETA"

    except Exception as e:
        result["status"] = "CRASH"
        result["issue"] = str(e)
        
    result["processing_time"] = round(time.time() - start_time, 4)
    return result

# --- 模式 1: 单文件调试 ---
def run_single_mode(input_file):
    print(f"==================================================")
    print(f"   Beta-Cooper CLI: Single File Mode")
    print(f"   Target: {os.path.basename(input_file)}")
    print(f"==================================================\n")

    res = analyze_structure(input_file, return_full_data=True)
    
    # 打印验证结果
    print(f"[Validator] Status:     {res['status']}")
    print(f"[Validator] Confidence: {res['confidence']:.2f}")
    if res['issue'] != 'None':
        print(f"[Validator] Issue:      {res['issue']}")
    
    print("-" * 30)
    
    # 打印几何结果
    if pd.notna(res['n_strands']):
        print(f"[Geometry] Strands (n): {int(res['n_strands'])}")
        print(f"[Geometry] Shear (S):   {int(res['shear_S'])}")
        print(f"[Geometry] Radius:      {res['radius']} Å")
        print(f"[Geometry] Tilt Angle:  {res['tilt']}°")
        print(f"[Geometry] Height:      {res['height']} Å")
        
        # 额外的调试信息
        if 'debug_geo' in res:
            geo = res['debug_geo']
            print(f"[Geometry] Keep Ratio:  {geo.audit.get('keep_ratio', 0)*100:.1f}%")
            if geo.audit.get('rescue_success'):
                print(f"[Geometry] NOTE: Rescue mechanism triggered and succeeded!")
    else:
        print("[Geometry] Skipped (No valid barrel detected)")

    print("-" * 30)
    print(f"Total Time: {res['processing_time']}s")

# --- 模式 2: 批量处理 ---
def run_batch_mode(input_dir, output_file, workers):
    extensions = ['*.pdb', '*.cif', '*.ent', '*.mmcif']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files = sorted(list(set([os.path.abspath(f) for f in files])))

    if not files:
        print(f"No structures found in {input_dir}")
        return

    print(f"🚀 Starting Batch Process on {len(files)} files ({workers} cores)...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(analyze_structure, f): f for f in files}
        
        iterator = tqdm(as_completed(future_to_file), total=len(files), unit="pdb", desc="Processing")
        for i, future in enumerate(iterator):
            try:
                data = future.result()
                results.append(data)
                
                # 每100个文件更新一次状态简报
                if (i + 1) % 100 == 0:
                    counts = {}
                    for r in results:
                        s = r.get('status', 'UNKNOWN')
                        counts[s] = counts.get(s, 0) + 1
                    summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    tqdm.write(f"[Progress] {summary}")
                    
            except Exception as e:
                tqdm.write(f"Error: {e}")

    # 保存结果
    df = pd.DataFrame(results)
    
    # 智能列排序
    preferred_order = [
        'id', 'status', 'confidence', 
        'n_strands', 'shear_S', 'radius', 'tilt', 'height', 
        'processing_time', 'issue'
    ]
    cols = [c for c in preferred_order if c in df.columns] + \
           [c for c in df.columns if c not in preferred_order]
    
    df = df[cols].sort_values(by=['status', 'id'])
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Batch completed! Saved to: {output_file}")
    print("--- Final Summary ---")
    print(df['status'].value_counts())
    print(f"Avg Time: {df['processing_time'].mean():.4f}s")

# --- 主入口 ---
def main():
    parser = argparse.ArgumentParser(description="Beta-Cooper: Beta-Barrel Analysis Tool")
    
    parser.add_argument("input", help="Path to a single PDB file OR a directory of PDBs")
    parser.add_argument("-o", "--output", default="barrel_census.csv", help="Output CSV path (Batch mode only)")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Number of CPU cores (Batch mode only)")
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.input)
    
    if not os.path.exists(input_path):
        print(f"Error: Input path not found: {input_path}")
        sys.exit(1)

    if os.path.isfile(input_path):
        # 自动进入单文件模式
        run_single_mode(input_path)
    elif os.path.isdir(input_path):
        # 自动进入批量模式
        run_batch_mode(input_path, args.output, args.workers)
    else:
        print("Error: Input is neither a file nor a directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()