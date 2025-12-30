# beta_cooper/validator/__init__.py

from .extractor import BarrelExtractor
from .analyzer import BarrelAnalyzer

class BarrelValidator:
    """
    BarrelValidator (Refactored)
    Facade class that orchestrates the validation pipeline:
    1. Extraction (via BarrelExtractor): PDB -> Segments & Coords
    2. Analysis (via BarrelAnalyzer): Geometry -> Metrics & Score
    """

    def __init__(self, pdb_file, chain_id='A'):
        self.pdb_file = pdb_file
        self.target_chain_id = chain_id 

    def validate(self):
        """
        Runs the full validation pipeline.
        Returns a dictionary with status, score, metrics, and debug data.
        """
        # --- Step 1: Extraction ---
        # 负责文件读取、OPM 清洗、DSSP 调用、Beta 片段提取
        extractor = BarrelExtractor(self.pdb_file, self.target_chain_id)
        segments, all_coords = extractor.run()

        # 如果提取失败（例如文件不存在、DSSP 报错），由 Analyzer 生成标准失败响应
        # 或者在这里直接处理。为了统一 metrics 格式，我们让 Analyzer 处理空数据报错。
        
        # --- Step 2: Analysis ---
        # 负责几何计算、异常值剔除、指标统计、打分
        analyzer = BarrelAnalyzer()
        result = analyzer.run(segments, all_coords)
        
        # --- Step 3: Combine Results ---
        # 将提取的原始数据 (debug_segments/coords) 注入到结果中
        # 这样上层应用 (如 batch_process.py) 可以拿去给 Geometry 模块使用
        result.update({
            "debug_segments": segments,
            "debug_coords": all_coords
        })
        
        return result