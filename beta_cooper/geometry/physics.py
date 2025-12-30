# beta_cooper/geometry/__init__.py

from .cleaner import BarrelCleaner
from .topology import BarrelTopology
from .physics import BarrelPhysics

class BarrelGeometry:
    """
    BarrelGeometry (Refactored)
    Facade class that orchestrates Cleaning, Topology, and Physics calculations.
    """

    def __init__(self, segments, all_coords, residue_ids=None):
        # 统一的审计/日志字典
        self.audit = {
            'status': 'OK',
            'keep_ratio': 0.0
        }
        
        # 1. Cleaning Phase (负责清洗、切割、坐标系对齐)
        cleaner = BarrelCleaner(self.audit)
        clean_segments, clean_coords = cleaner.run(segments, all_coords)
        
        # 计算保留率
        n_raw = len(all_coords)
        n_clean = len(clean_coords)
        self.audit['keep_ratio'] = round(n_clean / max(1, n_raw), 3)

        # 2. Topology Phase (负责连接片段、构建 Strands)
        topology = BarrelTopology(self.audit)
        
        if len(clean_coords) > 15:
            # 将清洗后的片段转换到桶坐标系 (apply_alignment) 后再进行连接
            aligned_segments = cleaner.apply_alignment(clean_segments)
            self.strands = topology.run(aligned_segments)
        else:
            self.strands = []
            if self.audit['status'] == 'OK':
                self.audit['status'] = 'Insufficient_Points_After_Purge'

        # 3. Physics Phase (负责计算几何参数)
        physics = BarrelPhysics(self.audit)
        self.params = physics.calculate(self.strands)
        
        # 将审计信息合并到最终参数中，方便查看
        self.params.update(self.audit)

    def get_summary(self):
        return self.params