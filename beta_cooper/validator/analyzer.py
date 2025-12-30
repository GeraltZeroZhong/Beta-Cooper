import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

class BarrelAnalyzer:
    """
    负责对提取出的 β-折叠原子坐标进行几何分析和质量评估。
    包含 PCA 分析、MCD 异常值检测和几何指标计算。
    """

    def __init__(self):
        pass

    def run(self, segments, all_coords):
        """
        执行分析流程。
        输入: segments (list of arrays), all_coords (N, 3 array)
        输出: 包含评分、状态和详细指标的字典
        """
        metrics = self._get_empty_metrics()
        
        # 1. 基础检查
        if segments is None or all_coords is None:
            return self._fail("FAIL_EXTRACT", "Extraction failed", metrics)

        metrics["n_segments"] = len(segments)
        metrics["n_atoms_total"] = len(all_coords)
        
        if len(all_coords) < 30:
            return self._fail("FAIL_LOW_BETA", f"Insufficient beta atoms ({len(all_coords)}<30)", metrics)

        # 2. PCA & 几何维度检查
        try:
            pca = PCA(n_components=3)
            centered_raw = all_coords - np.mean(all_coords, axis=0)
            aligned = pca.fit_transform(centered_raw)
            vars = pca.explained_variance_ 
            
            if vars[1] < 1e-5 or vars[2] < 1e-5: 
                return self._fail("FAIL_DEGENERATE", "Geometry is degenerate (2D/1D)", metrics)

            axis_dominance_ratio = vars[0] / vars[1]
            
            plane_idx = (1, 2)
            z_idx = 0
            plane_source = "PC1_Axis"
            
            # 如果主轴优势不明显 (接近球形)，尝试寻找最佳投影面
            if axis_dominance_ratio < 1.15:
                best_plane, best_cv = self._find_best_plane_fallback(aligned, vars, axis_dominance_ratio)
                if best_plane is None:
                    return self._fail("FAIL_AMBIGUOUS", "Shape ambiguous (Fallback failed)", metrics)
                plane_idx = best_plane
                z_idx = list(set([0, 1, 2]) - set(plane_idx))[0]
                plane_source = "Fallback_Search"
            
            metrics["plane_source"] = plane_source

            # 3. MCD 鲁棒内点选择 (Core Selection)
            pts_2d = aligned[:, plane_idx]
            
            mcd = MinCovDet(support_fraction=0.75, random_state=42).fit(pts_2d)
            dist_sq = mcd.mahalanobis(pts_2d)
            threshold = np.percentile(dist_sq, 80)
            inlier_mask = dist_sq <= threshold
            pts_core = pts_2d[inlier_mask]
            
            metrics["n_atoms_core"] = len(pts_core)
            metrics["inlier_ratio"] = len(pts_core) / len(pts_2d) if len(pts_2d) > 0 else 0.0
            
            if len(pts_core) < 15:
                return self._fail("FAIL_UNSTABLE", "Too few core atoms (<15)", metrics)

            center = np.median(pts_core, axis=0)
            is_small_sample = len(pts_core) < 30 
            
            # 4. 详细指标计算
            core_cov = np.cov(pts_core.T)
            evals, _ = np.linalg.eigh(core_cov)
            min_eval = max(evals[0], 1e-9)
            cov_eigen_ratio = evals[1] / min_eval
            ellipticity = np.sqrt(cov_eigen_ratio)
            
            radii = np.linalg.norm(pts_core - center, axis=1)
            avg_r = np.mean(radii) + 1e-6
            ring_cv = np.std(radii) / avg_r
            
            rad_kurtosis = 0.0 if is_small_sample else self._calc_kurtosis_safe(radii)
            
            z_core = aligned[inlier_mask, z_idx]
            local_thickness = self._calc_local_thickness_spatial(radii, z_core)
            
            # 角度覆盖度检查 (Angular Gap)
            dx = pts_core[:, 0] - center[0]
            dy = pts_core[:, 1] - center[1]
            angles = np.degrees(np.arctan2(dy, dx)) % 360
            
            n_bins = 18 if is_small_sample else 36
            hist, _ = np.histogram(angles, bins=n_bins, range=(0, 360))
            hist_doubled = np.concatenate([hist, hist])
            max_zeros = 0
            curr_zeros = 0
            for c in hist_doubled:
                if c == 0: curr_zeros += 1
                else: 
                    max_zeros = max(max_zeros, curr_zeros)
                    curr_zeros = 0
            max_zeros = max(max_zeros, curr_zeros)
            max_zeros = min(max_zeros, n_bins)
            angular_gap = max_zeros * (360.0 / n_bins)
            
            # Z轴轮廓稳定性
            try:
                sort_idx = np.argsort(z_core)
                r_sorted = radii[sort_idx]
                n_chunks = max(2, min(4, len(z_core) // 15))
                chunks = np.array_split(r_sorted, n_chunks)
                chunk_means = [np.mean(c) for c in chunks if len(c) > 0]
                z_profile_cv = np.std(chunk_means)/(np.mean(chunk_means)+1e-6) if len(chunk_means)>1 else 0.0
            except:
                z_profile_cv = 1.0

            metrics.update({
                "ellipticity": ellipticity,
                "cov_eigen_ratio": cov_eigen_ratio,
                "ring_cv": ring_cv,
                "rad_kurtosis": rad_kurtosis,
                "local_thickness": local_thickness,
                "angular_gap": angular_gap,
                "n_angular_bins": n_bins,
                "z_profile_cv": z_profile_cv
            })

            # 5. 评分与惩罚 (Scoring)
            return self._calculate_score(metrics, is_small_sample, plane_source)

        except Exception as e:
             return self._fail("FAIL_MATH", f"Math Error: {str(e)}", metrics)

    def _calculate_score(self, metrics, is_small_sample, plane_source):
        """应用惩罚项计算最终置信度"""
        ellipticity = metrics["ellipticity"]
        cov_eigen_ratio = metrics["cov_eigen_ratio"]
        local_thickness = metrics["local_thickness"]
        rad_kurtosis = metrics["rad_kurtosis"]
        angular_gap = metrics["angular_gap"]
        z_profile_cv = metrics["z_profile_cv"]
        n_segments = metrics["n_segments"]
        ring_cv = metrics["ring_cv"]

        # 计算惩罚
        if ellipticity > 1.8: metrics["pen_ellipticity"] = (ellipticity - 1.8) * 0.6
        if cov_eigen_ratio > 50: metrics["pen_stability"] = min(0.3, 0.05 * np.log10(cov_eigen_ratio / 50))
        
        # [RESTORED] Ring CV Penalty (严格控制圆度)
        cv_thr = 0.20 if is_small_sample else 0.25
        cv_mult = 3.5 if is_small_sample else 2.5
        if ring_cv > cv_thr: 
            metrics["pen_ringcv"] = (ring_cv - cv_thr) * cv_mult
        
        # [RESTORED] Thickness Penalty (严格控制厚度/空心度)
        thick_thr = 0.25 if is_small_sample else 0.30
        thick_mult = 3.0 if is_small_sample else 2.5
        if local_thickness > thick_thr: 
            metrics["pen_thickness"] = (local_thickness - thick_thr) * thick_mult
        
        if not is_small_sample:
            if local_thickness > 0.35 and rad_kurtosis < -1.0: metrics["pen_kurtosis"] = 0.5 
            elif rad_kurtosis < -1.0: metrics["pen_kurtosis"] = 0.1
        
        if angular_gap > 60: metrics["pen_gap"] = (angular_gap - 60) / 100.0
        if z_profile_cv > 0.3: metrics["pen_z"] = 0.3
        
        if plane_source != "PC1_Axis": metrics["pen_fallback"] = 0.15
        if n_segments < 3: metrics["pen_segments"] = 0.25

        total_penalty = sum(v for k, v in metrics.items() if k.startswith("pen_"))
        score = max(0.0, 1.0 - total_penalty)
        
        # 判定是否有效 (Threshold > 0.6)
        is_valid = score > 0.6

        status = "OK" if is_valid else "FAIL_SCORE"
        issue = "None"
        if not is_valid:
            major = {k: v for k, v in metrics.items() if k.startswith("pen_") and v > 0.15}
            if major:
                max_k = max(major, key=major.get)
                issue = max_k.replace("pen_", "").upper() + "_ISSUE"
            else:
                issue = "LOW_CONFIDENCE"

        return {
            "status": status,
            "is_valid": is_valid,
            "confidence": score, 
            "issue": issue,
            "metrics": metrics
        }

    # --- 辅助计算方法 ---

    def _get_empty_metrics(self):
        return {
            "n_segments": np.nan, "n_atoms_total": np.nan, "n_atoms_core": np.nan,
            "inlier_ratio": np.nan, "n_angular_bins": np.nan, "plane_source": "NA",
            "ellipticity": np.nan, "cov_eigen_ratio": np.nan, "ring_cv": np.nan,
            "rad_kurtosis": np.nan, "local_thickness": np.nan, "angular_gap": np.nan,
            "z_profile_cv": np.nan, "pen_ellipticity": 0.0, "pen_stability": 0.0,
            "pen_ringcv": 0.0, "pen_thickness": 0.0, "pen_kurtosis": 0.0,
            "pen_gap": 0.0, "pen_z": 0.0, "pen_fallback": 0.0, "pen_segments": 0.0
        }

    def _fail(self, status, reason, metrics):
        return {"status": status, "is_valid": False, "confidence": 0.0, "issue": reason, "metrics": metrics}

    def _calc_kurtosis_safe(self, data):
        if len(data) < 30: return 0.0 
        mean = np.mean(data)
        diff = data - mean
        m4 = np.mean(diff**4)
        m2 = np.mean(diff**2)
        if m2 < 1e-6: return 0.0
        return (m4 / (m2**2)) - 3.0
        
    def _calc_local_thickness_spatial(self, radii, z_coords):
        try:
            z_quantiles = np.percentile(z_coords, [0, 25, 50, 75, 100])
            local_thicknesses = []
            for i in range(4):
                z_min, z_max = z_quantiles[i], z_quantiles[i+1]
                if i < 3: mask = (z_coords >= z_min) & (z_coords < z_max)
                else: mask = (z_coords >= z_min) & (z_coords <= z_max + 1e-9)
                chunk_r = radii[mask]
                if len(chunk_r) < 5: continue
                q75, q25 = np.percentile(chunk_r, [75, 25])
                median = np.median(chunk_r) + 1e-6
                local_thicknesses.append((q75 - q25) / median)
            if not local_thicknesses:
                q75, q25 = np.percentile(radii, [75, 25])
                return (q75 - q25) / (np.median(radii) + 1e-6)
            return np.median(local_thicknesses)
        except: return 1.0

    def _find_best_plane_fallback(self, aligned, all_vars, dom_ratio):
        best_cv = float('inf')
        best_plane = None
        if dom_ratio < 1.03: z_thresh_factor = 0.45
        elif dom_ratio < 1.08: z_thresh_factor = 0.60
        else: z_thresh_factor = 0.75
        for pair in [(1, 2), (0, 2), (0, 1)]:
            z_dim = list(set([0, 1, 2]) - set(pair))[0]
            z_var = all_vars[z_dim]
            plane_max_var = max(all_vars[pair[0]], all_vars[pair[1]])
            if z_var < plane_max_var * z_thresh_factor: continue
            pts = aligned[:, pair]
            try:
                mcd = MinCovDet(support_fraction=0.8, random_state=42).fit(pts)
                dist = mcd.mahalanobis(pts)
                thresh = np.percentile(dist, 80)
                mask = dist <= thresh
                if np.sum(mask) < 10: continue
                center = np.median(pts[mask], axis=0) 
                radii = np.linalg.norm(pts[mask] - center, axis=1)
                cv = np.std(radii) / (np.mean(radii) + 1e-6)
                if cv < best_cv:
                    best_cv = cv
                    best_plane = pair
            except: continue
        if best_cv > 0.5: return None, None 
        return best_plane, best_cv