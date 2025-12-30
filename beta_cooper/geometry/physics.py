import numpy as np

class BarrelPhysics:
    """
    负责计算 β-桶的物理几何参数 (Radius, Tilt, Shear, Height)。
    输入：有序的 Strands 列表。
    输出：包含物理参数的字典。
    """

    def __init__(self, audit_dict):
        self.audit = audit_dict

    def calculate(self, strands):
        """
        根据 Strands 的几何排列计算桶的参数。
        """
        # 默认空值
        params = {
            'n_strands': 0, 
            'radius': 0.0, 
            'tilt_angle': 0.0, 
            'shear_S': 0,
            'shear_S_raw': 0.0, 
            'height': 0.0
        }

        if not strands:
            return params

        n = len(strands)
        
        # 1. 准备数据：合并所有坐标
        # 注意：这里的 coords 已经是经过 Cleaner 对齐到桶坐标系 (Z轴为桶轴) 的
        all_local = np.vstack([s['coords'] for s in strands])
        
        # 2. 计算半径 (Radius)
        # 取所有 Cα 原子到 Z 轴距离的中位数
        dists = np.linalg.norm(all_local[:, :2], axis=1)
        R = np.median(dists) if len(dists) > 0 else 0.0

        # 3. 计算倾角 (Tilt Angle)
        # Strand 向量与 Z 轴夹角
        tilts = []
        for s in strands:
            vec = s['vector']
            # Z 轴是 (0,0,1)，点积即 vec[2]
            cos_theta = abs(vec[2])
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            tilts.append(angle)
        
        avg_tilt = np.clip(np.mean(tilts), 0, 85.0) if tilts else 0.0

        # 4. 计算剪切数 (Shear Number, S)
        # 使用 Murzin 理论公式近似
        a, b = 4.4, 3.3  # 典型的残基间距参数
        
        # 公式推导：(2*pi*R)^2 = (n*a)^2 + (S*b)^2 (Pythagoras on unrolled barrel)
        # => (S*b)^2 = (2*pi*R)^2 - (n*a)^2
        delta_sq = (2 * np.pi * R)**2 - (n * a)**2
        
        if delta_sq > -15.0: 
            # 正常情况：半径足够大，可以用毕达哥拉斯定理
            S_calc = np.sqrt(max(0.0, delta_sq)) / b
        else: 
            # 异常情况（比如桶极度倾斜或半径过小）：使用倾角回推
            # tan(alpha) = S*b / n*a
            S_calc = (n * a / b) * min(np.tan(np.radians(avg_tilt)), 5.0)

        # 5. 计算高度 (Height)
        z_coords = all_local[:, 2]
        height = np.max(z_coords) - np.min(z_coords) if len(z_coords) > 0 else 0.0

        params.update({
            "n_strands": n,
            "radius": round(R, 2),
            "tilt_angle": round(avg_tilt, 1),
            "shear_S_raw": round(S_calc, 2),
            # S 必须是偶数 (Shear Number 在封闭圆柱上通常为偶数，除非是 Mobius 带)
            # 这里取最近的偶数
            "shear_S": int(np.floor(S_calc / 2 + 0.5)) * 2,
            "height": round(height, 1)
        })

        return params
