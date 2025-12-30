import os
import sys
import glob
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 注意：我们不再需要 ProteinParser 了！
from beta_cooper.validator import BarrelValidator
from beta_cooper.geometry import BarrelGeometry

def process_single_structure(pdb_path):
    pdb_path = os.path.abspath(pdb_path)
    filename = os.path.basename(pdb_path)
    stem = Path(pdb_path).stem
    
    result = {
        "filename": filename,
        "id": stem,
        "status": "UNKNOWN",
        "confidence": 0.0,
        "n_strands": np.nan,
        "shear_S": np.nan,
        "radius": np.nan,
        "tilt": np.nan,
        "height": np.nan,
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    try:
        # --- 1. Validate & Extract ---
        # Validator V25 now does EVERYTHING: Sanitization, Extraction, Validation
        validator = BarrelValidator(pdb_path)
        v_res = validator.validate()
        
        result.update({
            "status": v_res['status'],
            "confidence": v_res['confidence'],
            "issue": v_res['issue'],
            **v_res['metrics']
        })
        
        # If segments were extracted (even if validation failed score-wise),
        # we can optionally still try to calculate geometry, but typically
        # we only care if it's Valid.
        # But let's trust the 'is_valid' flag.
        
        if not v_res['is_valid']:
            result["processing_time"] = time.time() - start_time
            return result

        # --- 2. Geometry Calculation ---
        # CRITICAL UPDATE: Don't re-parse! Use what Validator found.
        beta_segments = v_res.get('debug_segments')
        all_coords = v_res.get('debug_coords')

        if not beta_segments or len(beta_segments) == 0:
            result["status"] = "FAIL_NO_BETA"
            return result

        # Calculate Physics using the clean segments
        geo = BarrelGeometry(segments=beta_segments, all_coords=all_coords)
        params = geo.get_summary()
        
        result.update({
            "n_strands": params['n_strands'],
            "shear_S": params['shear_S'],
            "radius": params['radius'],
            "tilt": params['tilt_angle'],
            "height": params['height']
        })
        
    except Exception as e:
        result["status"] = "CRASH"
        result["issue"] = str(e)
        
    result["processing_time"] = time.time() - start_time
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Beta-Cooper Batch Harvester")
    parser.add_argument("--input", "-i", required=True, help="Folder containing PDB/CIF files")
    parser.add_argument("--output", "-o", default="barrel_census.csv", help="Output CSV file")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Cores")
    args = parser.parse_args()

    input_dir = args.input
    output_file = args.output
    
    extensions = ['*.pdb', '*.cif', '*.ent']
    files = []
    for ext in extensions:
        raw_files = glob.glob(os.path.join(input_dir, ext))
        files.extend([os.path.abspath(f) for f in raw_files])
    
    if not files:
        print(f"No structures found in {input_dir}")
        return

    print(f"🚀 Starting Harvest on {len(files)} structures using {args.workers} cores...")
    
    results = []
    
    if args.workers == 1:
        for i, f in enumerate(files):
            data = process_single_structure(f)
            results.append(data)
            if (i+1) % 10 == 0:
                print(f"[{((i+1)/len(files))*100:.1f}%] {os.path.basename(f)} -> {data['status']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_single_structure, f): f for f in files}
            total = len(files)
            completed = 0
            for future in as_completed(future_to_file):
                data = future.result()
                results.append(data)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    percent = (completed / total) * 100
                    print(f"[{percent:.1f}%] Processed {completed}/{total} - Last: {data['filename']} ({data['status']})")

    df = pd.DataFrame(results)
    cols = ['id', 'status', 'confidence', 'n_strands', 'shear_S', 'radius', 'tilt', 'issue']
    existing_cols = [c for c in cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining]
    
    df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Census saved to: {output_file}")
    print("\nSummary:")
    print(df['status'].value_counts())

if __name__ == "__main__":
    main()