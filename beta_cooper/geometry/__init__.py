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
        # 1. 初始化审计/日志字典
        self.audit = {
            'status': 'OK',
            'keep_ratio': 0.0,
            # 保留原有的 audit 字段默认值，防止外部调用报错（如果外部依赖某些特定key）
            'n_splits': 0, 'n_merges': 0, 'n_flips': 0, 
            'n_dropped_short': 0, 'n_linear_splits': 0,
            'n_rejected_helices': 0, 'n_rejected_outliers': 0,
            'rescue_triggered': False, 'rescue_success': False,
            'parity_method': 'None', 'dbscan_clusters_kept': 0
        }
        
        # 2. 清洗阶段 (Cleaning Phase)
        # 负责：数据清洗、DBSCAN聚类、初始切割、坐标系对齐计算
        cleaner = BarrelCleaner(self.audit)
        self.clean_segments, self.clean_coords = cleaner.run(segments, all_coords)
        
        # 获取对齐参数供后续使用
        self.rotation_matrix = cleaner.rotation_matrix
        self.centroid = cleaner.centroid
        
        # 计算保留率
        n_raw = len(all_coords)
        n_clean = len(self.clean_coords)
        self.audit['keep_ratio'] = round(n_clean / max(1, n_raw), 3)

        # 3. 拓扑阶段 (Topology Phase)
        # 负责：片段连接、方向调整、奇偶性修复、救援模式
        topology = BarrelTopology(self.audit)
        
        if len(self.clean_coords) > 15:
            # 关键步骤：将清洗后的片段转换到“桶坐标系”（Z轴为轴心）再进行拓扑连接
            self.aligned_segments = cleaner.apply_alignment(self.clean_segments)
            self.strands = topology.run(self.aligned_segments)
        else:
            self.aligned_segments = []
            self.strands = []
            if self.audit['status'] == 'OK':
                self.audit['status'] = 'Insufficient_Points_After_Purge'

        # 4. 物理参数阶段 (Physics Phase)
        # 负责：计算 Radius, Tilt, Shear, Height
        physics = BarrelPhysics(self.audit)
        self.params = physics.calculate(self.strands)
        
        # 将审计信息合并到最终参数中，确保输出包含所有调试信息
        self.params.update(self.audit)

    def get_summary(self):
        """返回包含物理参数和审计信息的字典"""
        return self.params