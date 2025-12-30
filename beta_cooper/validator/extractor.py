import os
import shutil
import subprocess
import tempfile
import numpy as np
from Bio.PDB import PDBParser

class BarrelExtractor:
    """
    负责从 PDB 文件中提取 β-折叠片段 (Segments)。
    包含 OPM 格式清洗、DSSP 调用和链过滤逻辑。
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
        """
        执行提取流程。
        返回: (segments, all_coords)
        如果失败返回 (None, None)
        """
        if not self.file_exists:
            return None, None

        # 查找 DSSP 可执行文件
        dssp_bin = shutil.which("mkdssp") or shutil.which("dssp") or "/usr/bin/mkdssp"
        if not os.path.exists(dssp_bin): 
            # 如果找不到 DSSP，这是一个严重的环境问题，但在代码层面返回 None 表示无法提取
            return None, None
        
        pdb_abspath = os.path.abspath(self.pdb_file)
        
        # 1. Sanitize (清洗 PDB 文件，特别是针对 OPM 格式)
        # 使用临时文件避免修改原文件
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_in:
            sanitized_path = tmp_in.name
            try:
                self._sanitize_opm(pdb_abspath, tmp_in)
            except Exception:
                tmp_in.close()
                if os.path.exists(sanitized_path): os.remove(sanitized_path)
                return None, None
        
        # 2. Run DSSP (Legacy Execution)
        # 将清洗后的结构传给 DSSP 计算二级结构
        with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp_out:
            output_file_path = tmp_out.name
            
        try:
            cmd = [dssp_bin, sanitized_path, output_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0: 
                return None, None
                
            dssp_data = self._parse_dssp_raw(output_file_path)
            
        except Exception: 
            return None, None
        finally:
            # 清理临时 DSSP 文件
            if os.path.exists(output_file_path): os.remove(output_file_path)

        # 3. Reload & Map (重新加载结构并映射 DSSP 结果)
        clean_parser = PDBParser(QUIET=True)
        try:
            clean_struct = clean_parser.get_structure('clean', sanitized_path)
            clean_model = clean_struct[0]
        except Exception:
            if os.path.exists(sanitized_path): os.remove(sanitized_path)
            return None, None
            
        # 清理临时 PDB 文件
        if os.path.exists(sanitized_path): os.remove(sanitized_path)

        # 确定目标链
        target = self.target_chain_id if self.target_chain_id and self.target_chain_id.strip() else 'A'
        if target in clean_model: 
            chain_obj = clean_model[target]
        elif 'A' in clean_model: 
            chain_obj = clean_model['A']
        else: 
            # 如果指定链和A链都不存在，回退到第一条链
            chain_obj = list(clean_model)[0]
        
        # 提取片段
        segments = []
        current_segment = []
        last_resseq = -999
        JUMP_TOL = 4 # 允许的最大残基编号跳跃，超过即视为断开
        
        for residue in chain_obj:
            if not residue.has_id('CA'): continue
            
            c_id = chain_obj.id
            r_seq = residue.id[1]
            
            # 从 DSSP 字典中获取二级结构
            ss = dssp_data.get((c_id, r_seq))
            if ss is None:
                # 尝试容错匹配（空链名或 A 链混用）
                if c_id == 'A': ss = dssp_data.get((' ', r_seq))
                elif c_id == ' ': ss = dssp_data.get(('A', r_seq))
            
            if ss == 'E': # 'E' 代表 Beta-Sheet
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
        
        # 处理最后一段
        if len(current_segment) >= 3:
            segments.append(np.array(current_segment))
            
        all_coords = np.concatenate(segments) if segments else np.array([])
        
        return segments, all_coords

    def _parse_dssp_raw(self, dssp_file):
        """解析 DSSP 输出文件，提取二级结构信息"""
        data = {}
        start = False
        with open(dssp_file, 'r') as f:
            for line in f:
                if "  #  RESIDUE" in line:
                    start = True
                    continue
                if not start: continue
                if len(line) < 20: continue
                try:
                    res_seq_str = line[5:10].strip()
                    if not res_seq_str: continue 
                    res_seq = int(res_seq_str)
                    
                    chain_id = line[11].strip()
                    if not chain_id: chain_id = ' '
                    
                    ss = line[16]
                    if ss == ' ': ss = '-'
                    
                    data[(chain_id, res_seq)] = ss
                except ValueError: continue
        return data

    def _sanitize_opm(self, input_path, output_handle):
        """
        清洗 OPM 数据库的 PDB 文件。
        OPM 文件有时包含特殊的 DUM 原子或非标准格式，可能导致 Biopython/DSSP 崩溃。
        """
        output_handle.write("HEADER    OPM CLEANED PDB\n")
        # 写入一个标准的晶胞参数，防止某些解析器报错
        output_handle.write("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
        
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"): 
                    res_name = line[17:20]
                    # 过滤掉 OPM 的虚拟原子 (DUM) 和水 (HOH)
                    if res_name == "DUM" or res_name == "HOH": continue
                    
                    # 修复某些旧格式 PDB 的链标识符位置
                    if len(line) > 21 and line[21] == ' ':
                        line = line[:21] + 'A' + line[22:]
                    output_handle.write(line)
                elif line.startswith("TER") or line.startswith("END"):
                     output_handle.write(line)
        output_handle.flush()