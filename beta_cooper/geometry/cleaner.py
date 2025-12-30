import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class BarrelCleaner:
    """
    负责原始坐标的数据清洗、DBSCAN 聚类筛选、片段切割 (Purify) 以及坐标系对齐。
    """

    def __init__(self, audit_dict):
        # 引用传入的 audit 字典，以便记录清洗过程中的统计信息
        self.audit = audit_dict
        
        # 初始化相关的 audit 计数（如果尚未存在）
        defaults = {
            'n_splits': 0, 'n_dropped_short': 0, 
            'n_rejected_helices': 0, 'n_rejected_outliers': 0,
            'dbscan_clusters_kept': 0, 'status': 'OK'
        }
        for k, v in defaults.items():
            if k not in self.audit:
                self.audit[k] = v

        # 对齐参数
        self.rotation_matrix = np.eye(3)
        self.centroid = np.zeros(3)

    def run(self, segments, all_coords):
        """
        执行完整的清洗和对齐流程。
        返回: (clean_segments, clean_coords)
        注意: 返回的坐标还是原始空间的，需调用 apply_alignment 转换到桶坐标系。
        """
        # 1. 标准化 (Standardization)
        # 如果没有传入片段但有坐标，或者片段列表为空，尝试自动分割
        if not segments and len(all_coords) > 0:
            initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)
        else:
            initial_segments = segments if segments else []
            if not initial_segments and len(all_coords) > 0:
                initial_segments = self._auto_segment_coords(all_coords, gap_threshold=4.5)

        # 2. 清洗管道 (The Purge)
        clean_segments, clean_coords = self._purify_pipeline_v23(initial_segments)
        
        # 3. 计算对齐参数 (Alignment Calculation)
        if len(clean_coords) > 15:
            self._calculate_alignment(clean_coords)
        else:
            # 点太少，无法可靠对齐，保持原样
            self.centroid = np.mean(clean_coords, axis=0) if len(clean_coords) > 0 else np.zeros(3)
            self.rotation_matrix = np.eye(3)

        return clean_segments, clean_coords

    def apply_alignment(self, segments):
        """
        将片段从原始 PDB 坐标系转换到对齐后的桶坐标系 (Z轴为桶轴)。
        """
        aligned_segments = []
        for seg in segments:
            # (N, 3) @ (3, 3).T -> (N, 3)
            seg_local = (seg - self.centroid) @ self.rotation_matrix.T
            aligned_segments.append(seg_local)
        return aligned_segments

    def _auto_segment_coords(self, coords, gap_threshold=4.5):
        segments = []
        if len(coords) == 0: return []
        current_seg = [coords[0]]
        for i in range(1, len(coords)):
            if np.linalg.norm(coords[i] - coords[i-1]) > gap_threshold:
                if len(current_seg) >= 3: 
                    segments.append(np.array(current_seg))
                else:
                    self.audit['n_dropped_short'] += 1
                current_seg = []
            current_seg.append(coords[i])
        if len(current_seg) >= 3: 
            segments.append(np.array(current_seg))
        else:
            self.audit['n_dropped_short'] += 1
        return segments

    def _purify_pipeline_v23(self, initial_segments):
        """
        V23 Update: Multi-Extraction per segment.
        Extracts ALL valid sub-segments belonging to kept clusters.
        """
        if not initial_segments: return [], np.array([])
        
        # 1. Split (基于几何扭结的初步切割)
        splitted_segments = []
        for seg in initial_segments:
            sub_segs = self._scan_and_split(seg, cos_threshold=-0.2)
            if len(sub_segs) > 1: self.audit['n_splits'] += (len(sub_segs) - 1)
            splitted_segments.extend(sub_segs)

        # 2. Filter Helices (过滤掉过于笔直的螺旋结构)
        candidates = []
        for seg in splitted_segments:
            if len(seg) < 3: 
                self.audit['n_dropped_short'] += 1
                continue
            
            v_start = seg[-1] - seg[0]
            len_direct = np.linalg.norm(v_start)
            len_path = np.sum([np.linalg.norm(seg[k] - seg[k-1]) for k in range(1, len(seg))])
            linearity = len_direct / (len_path + 1e-6)
            
            # 简单的 PCA 检查以排除极端的线性结构
            pca = PCA(n_components=min(3, len(seg)))
            pca.fit(seg)
            residual = np.sqrt(np.sum(pca.explained_variance_[1:])) if len(seg) > 3 else 0.0
            
            keep = False
            if linearity > 0.80: keep = True  # 很直，可能是 Strand
            elif linearity > 0.60 and residual < 2.5: keep = True # 稍微弯曲但很细
            
            if keep: candidates.append(seg)
            else: self.audit['n_rejected_helices'] += 1
        
        if not candidates: return [], np.array([])
        
        # 3. DBSCAN (Dual Cluster + Multi-Extract)
        all_candidate_coords = np.vstack(candidates)
        try:
            clustering = DBSCAN(eps=6.5, min_samples=4).fit(all_candidate_coords)
            labels = clustering.labels_
            unique_labels = list(set(labels) - {-1})
            
            if not unique_labels: return [], np.array([])
            
            # Select Clusters (保留最大的1-2个簇)
            counts = {l: np.sum(labels == l) for l in unique_labels}
            sorted_labels = sorted(unique_labels, key=lambda l: counts[l], reverse=True)
            
            valid_labels = {sorted_labels[0]}
            self.audit['dbscan_clusters_kept'] = 1
            if len(sorted_labels) > 1:
                l1, l2 = sorted_labels[0], sorted_labels[1]
                if counts[l2] >= 0.4 * counts[l1]:
                    valid_labels.add(l2)
                    self.audit['dbscan_clusters_kept'] = 2

            final_segments = []
            cursor = 0
            for seg in candidates:
                seg_len = len(seg)
                seg_labels = labels[cursor : cursor + seg_len]
                cursor += seg_len
                
                # V23 Fix: Allow extracting multiple sub-segments from different clusters
                kept_mask = np.zeros(seg_len, dtype=bool)
                new_subsegs = []

                for lab in valid_labels:
                    idx = np.where(seg_labels == lab)[0]
                    if len(idx) < 3: continue
                    
                    split_locs = np.where(np.diff(idx) != 1)[0] + 1
                    groups = np.split(idx, split_locs)
                    # 只取该 Label 下最长的一段
                    g = max(groups, key=len)
                    
                    if len(g) < 3: continue
                    
                    s, e = int(g[0]), int(g[-1])
                    new_subsegs.append(seg[s:e+1])
                    kept_mask[s:e+1] = True
                
                if not new_subsegs:
                    self.audit['n_rejected_outliers'] += seg_len
                    continue
                
                # Correct outlier counting
                self.audit['n_rejected_outliers'] += int(seg_len - kept_mask.sum())
                final_segments.extend(new_subsegs)
                
            final_coords = np.vstack(final_segments) if final_segments else np.array([])
            return final_segments, final_coords

        except Exception:
            self.audit['status'] = 'DBSCAN_Fallback'
            # 如果 DBSCAN 失败，返回所有候选
            return candidates, all_candidate_coords

    def _scan_and_split(self, seg, cos_threshold=-0.2):
        """
        基于余弦相似度检测急剧转弯 (Kink) 并进行切割。
        """
        min_seg_len = 4
        if len(seg) < min_seg_len * 2: return [seg]
        
        # 平滑处理以减少噪声
        seg_smooth = seg.copy()
        for k in range(3):
            seg_smooth[1:-1, k] = (seg[:-2, k] + seg[1:-1, k] + seg[2:, k]) / 3.0
            
        window = 3
        limit = len(seg) - 2 * window
        best_idx, min_cos = -1, 1.0
        
        for i in range(0, limit):
            p0, p1, p2 = seg_smooth[i], seg_smooth[i+window], seg_smooth[i+2*window]
            v1, v2 = p1-p0, p2-p1
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 0.1 or n2 < 0.1: continue
            
            cos_val = np.dot(v1, v2) / (n1 * n2)
            if cos_val < cos_threshold:
                if cos_val < min_cos:
                    cand = i + window
                    if cand >= min_seg_len and (len(seg)-cand) >= min_seg_len:
                        min_cos, best_idx = cos_val, cand
                        
        if best_idx != -1:
            return self._scan_and_split(seg[:best_idx], cos_threshold) + \
                   self._scan_and_split(seg[best_idx:], cos_threshold)
        return [seg]

    def _calculate_alignment(self, coords):
        """
        计算主轴并生成旋转矩阵，使桶轴沿 Z 轴对齐。
        """
        self.centroid = np.mean(coords, axis=0)
        centered = coords - self.centroid
        pca = PCA(n_components=3)
        pca.fit(centered)
        basis = pca.components_
        vars = pca.explained_variance_
        
        # 寻找方差最接近的两个轴 (即圆形的横截面)，剩下的那个就是 Z 轴 (桶轴)
        candidates = []
        eps = 1e-9
        for i in range(3):
            others = [k for k in range(3) if k != i]
            v1, v2 = vars[others[0]], vars[others[1]]
            # 比较另外两个轴的方差比率（log ratio 越接近 0 说明越接近圆形）
            metric = abs(np.log((v1 + eps) / (v2 + eps)))
            candidates.append((i, metric))
        
        candidates.sort(key=lambda x: x[1])
        best_axis_idx = candidates[0][0]
        
        z_vec = basis[best_axis_idx]
        others = [k for k in range(3) if k != best_axis_idx]
        x_vec = basis[others[0]]
        
        # Gram-Schmidt 正交化确保完美正交
        z_vec /= np.linalg.norm(z_vec)
        x_vec /= np.linalg.norm(x_vec)
        x_vec = x_vec - np.dot(x_vec, z_vec) * z_vec
        x_vec /= np.linalg.norm(x_vec)
        y_vec = np.cross(z_vec, x_vec)
        
        self.rotation_matrix = np.vstack([x_vec, y_vec, z_vec])