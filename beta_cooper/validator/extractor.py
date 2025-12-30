import os
import shutil
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser, PDBIO
# 精确导入类
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio import BiopythonWarning

# 忽略 Biopython 非致命警告
warnings.simplefilter('ignore', BiopythonWarning)

class BarrelExtractor:
    """
    负责从 PDB 文件中提取 β-折叠片段。
    流程：
    1. Sanitize: 清洗 DUM/HOH 原子，修复空链 ID。
    2. Merge: 将所有链合并为单链 A，并强制写入 DSSP 所需的 HEADER。
    3. DSSP: 运行 mkdssp 计算二级结构。
    4. Extract: 映射结果并提取坐标。
    """

    def __init__(self, pdb_file, target_chain_id='A'):
        self.pdb_file = pdb_file
        self.target_chain_id = target_chain_id 
        self.file_exists = False
        try:
            with open(pdb_file, 'r'): pass
            self.file_exists = True
        except:
            self.file_exists = False

    def run(self):
        if not self.file_exists: return None, None

        # 1. 检查 DSSP 工具
        dssp_bin = shutil.which("mkdssp") or shutil.which("dssp") or "/usr/bin/mkdssp"
        if not dssp_bin or not os.path.exists(dssp_bin): 
            print(f"DEBUG: DSSP binary not found! Please install 'dssp' or 'mkdssp'.")
            return None, None
        
        pdb_abspath = os.path.abspath(self.pdb_file)
        
        # 临时文件路径
        sanitized_path = None
        merged_path = None
        output_dssp_path = None

        try:
            # --- Step 1: Sanitize (清洗原始文本) ---
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_in:
                sanitized_path = tmp_in.name
                try:
                    self._sanitize_opm(pdb_abspath, tmp_in)
                except Exception as e:
                    print(f"DEBUG: Sanitize failed: {e}")
                    return None, None
            
            # --- Step 2: Merge Chains (合并链 + 补充 Header) ---
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_merged:
                merged_path = tmp_merged.name
            
            if not self._merge_chains_to_A(sanitized_path, merged_path):
                print(f"DEBUG: Merge chains failed for {self.pdb_file}")
                return None, None

            # --- Step 3: Run DSSP ---
            with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp_out:
                output_dssp_path = tmp_out.name
            
            cmd = [dssp_bin, merged_path, output_dssp_path]
            
            try:
                # [FIX] 设置超时为 5秒，快速跳过复杂文件
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode != 0: 
                    # 只有非超时错误才打印详细日志
                    print(f"DEBUG: DSSP failed (Code {result.returncode}). Stderr: {result.stderr.strip()[:100]}...")
                    return None, None
            
            except subprocess.TimeoutExpired:
                # [FIX] 捕获超时，不抛出异常，只记录并跳过
                print(f"DEBUG: DSSP timed out (>5s) for {os.path.basename(self.pdb_file)}. Skipping.")
                return None, None
                
            dssp_data = self._parse_dssp_raw(output_dssp_path)

            # --- Step 4: Reload & Map ---
            clean_parser = PDBParser(QUIET=True)
            clean_struct = clean_parser.get_structure('merged', merged_path)
            clean_model = clean_struct[0]
            
            if 'A' not in clean_model: return None, None
            chain_obj = clean_model['A']

            # --- Step 5: Extract Segments ---
            segments = []
            current_segment = []
            last_resseq = -999
            JUMP_TOL = 4 
            
            for residue in chain_obj:
                if not residue.has_id('CA'): continue
                
                c_id = 'A'
                r_seq = residue.id[1]
                
                ss = dssp_data.get((c_id, r_seq), '-')
                
                if ss == 'E': 
                    if abs(r_seq - last_resseq) > JUMP_TOL:
                        if len(current_segment) >= 3:
                            segments.append(np.array(current_segment))
                        current_segment = []
                    current_segment.append(residue['CA'].get_coord())
                    last_resseq = r_seq
                else:
                    if len(current_segment) >= 3:
                        segments.append(np.array(current_segment))
                    current_segment = []
            
            if len(current_segment) >= 3:
                segments.append(np.array(current_segment))
                
            all_coords = np.concatenate(segments) if segments else np.array([])
            return segments, all_coords

        except Exception as e: 
            # 捕获其他未知错误
            print(f"DEBUG: Global Error parsing {self.pdb_file}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
        finally:
            # 清理临时文件
            for path in [sanitized_path, merged_path, output_dssp_path]:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except: pass

    def _merge_chains_to_A(self, input_pdb, output_pdb):
        """
        读取清洗后的 PDB，将所有链合并为 Chain A。
        强制写入 HEADER 以满足 DSSP 要求。
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('temp', input_pdb)
            if not structure: return False
            model = structure[0]
        except Exception: return False

        new_chain = Chain('A')
        res_counter = 1
        
        for chain in model:
            for residue in chain:
                residue.id = (residue.id[0], res_counter, ' ')
                new_chain.add(residue)
                res_counter += 1

        new_model = Model(0)
        new_model.add(new_chain)
        new_struct = Structure('merged')
        new_struct.add(new_model)

        io = PDBIO()
        io.set_structure(new_struct)
        
        try:
            with open(output_pdb, 'w') as f:
                # 写入 DSSP 必需的头信息
                f.write("HEADER    MERGED CHAIN A FOR DSSP\n")
                f.write("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
                io.save(f)
            return True
        except Exception: return False

    def _sanitize_opm(self, input_path, output_handle):
        """
        仅负责清洗原子数据，不再写入冗余 Header。
        """
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"): 
                    res_name = line[17:20]
                    if res_name == "DUM" or res_name == "HOH": continue
                    
                    if len(line) > 21 and line[21] == ' ':
                        line = line[:21] + 'A' + line[22:]
                    
                    output_handle.write(line)
                
                elif line.startswith("TER") or line.startswith("END"):
                     output_handle.write(line)
        output_handle.flush()

    def _parse_dssp_raw(self, dssp_file):
        """解析 DSSP 输出"""
        data = {}
        start = False
        with open(dssp_file, 'r') as f:
            for line in f:
                if "  #  RESIDUE" in line:
                    start = True; continue
                if not start: continue
                if len(line) < 20: continue
                try:
                    res_seq = int(line[5:10].strip())
                    ss = line[16]
                    if ss == ' ': ss = '-'
                    data[('A', res_seq)] = ss
                except ValueError: continue
        return data