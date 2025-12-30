import numpy as np

class BarrelTopology:
    """
    负责片段的拓扑连接、链的配对、奇偶性修复及低点数救援 (Rescue)。
    输入：对齐后的片段 (Aligned Segments)。
    输出：有序的 Strands 列表 (List of dicts)。
    """

    def __init__(self, audit_dict):
        self.audit = audit_dict
        
        # 初始化相关的 audit 计数
        defaults = {
            'n_merges': 0, 'n_flips': 0, 'n_linear_splits': 0,
            'rescue_triggered': False, 'rescue_success': False,
            'parity_method': 'None'
        }
        for k, v in defaults.items():
            if k not in self.audit:
                self.audit[k] = v

        # 匹配参数配置
        self.match_cfg = {
            'MAX_DIST': 4.5,
            'MIN_COS': 0.5,
            'MAX_Z_DIFF': 6.0,
            'MAX_THETA_DIFF': 1.2,
            'FLIP_MARGIN': 0.5,
            'R_GATE': 2.0,       
            'MAX_R_DIFF': 3.0    
        }
        
        self.strands = []

    def run(self, aligned_segments):
        """
        执行拓扑构建主流程。
        """
        if not aligned_segments:
            return []

        # 1. 初始拓扑构建
        self.strands = self._refine_topology(pool_source=aligned_segments)
        
        # 2. 救援机制 (Under-count Rescue)
        # 如果构建出的 Strands 数量过少 (<=6)，尝试更细粒度的切割并重新构建
        if 0 < len(self.strands) <= 6:
            self._rescue_under_count(aligned_segments)
            
        # 3. 最终整理
        # 再次尝试分解复杂的 Strands，并确保数量为偶数
        self._decompose_complex_strands()
        if len(self.strands) % 2 != 0:
            self._fix_parity()
            
        return self.strands

    def _refine_topology(self, pool_source, cfg=None):
        """
        核心贪婪算法：将片段池连接成最长的链。
        """
        active_pool = []
        for i, seg in enumerate(pool_source):
            if len(seg) < 3: continue
            head_anchor = self._endpoint_anchor(seg, 'head', w=4)
            tail_anchor = self._endpoint_anchor(seg, 'tail', w=4)
            active_pool.append({
                'coords': seg,
                'head': head_anchor,
                'tail': tail_anchor,
                'vec': self._get_strand_vector(seg),
                'used': False,
                'id': i
            })

        merged_strands = []
        while True:
            # 选取种子 (Seed)
            seed = None
            for s in active_pool:
                if not s['used']:
                    seed = s
                    break
            if not seed: break
            seed['used'] = True
            
            # 向前搜索 (Forward)
            curr = seed
            fwd_chain = []
            while True:
                match = self._find_best_match(curr, active_pool, direction='forward', cfg=cfg)
                if match:
                    cand, is_flip = match
                    if is_flip: self._flip_segment(cand)
                    cand['used'] = True
                    fwd_chain.append(cand)
                    curr = cand
                    self.audit['n_merges'] += 1
                else: break
            
            # 向后搜索 (Backward)
            curr = seed
            bwd_chain = []
            while True:
                match = self._find_best_match(curr, active_pool, direction='backward', cfg=cfg)
                if match:
                    cand, is_flip = match
                    if is_flip: self._flip_segment(cand)
                    cand['used'] = True
                    bwd_chain.append(cand)
                    curr = cand
                    self.audit['n_merges'] += 1
                else: break
            
            # 合并链条
            full_chain = bwd_chain[::-1] + [seed] + fwd_chain
            combined_coords = np.vstack([x['coords'] for x in full_chain])
            merged_strands.append({
                'coords': combined_coords,
                'vector': self._get_strand_vector(combined_coords)
            })

        return merged_strands

    def _find_best_match(self, curr, pool, direction, cfg=None):
        """
        在池中寻找与当前片段最匹配的下一个片段。
        """
        cfg = self.match_cfg if cfg is None else cfg
        MAX_DIST = cfg['MAX_DIST']
        MIN_COS = cfg['MIN_COS']
        MAX_Z_DIFF = cfg['MAX_Z_DIFF']
        MAX_THETA_DIFF = cfg['MAX_THETA_DIFF']
        FLIP_MARGIN = cfg['FLIP_MARGIN']
        R_GATE = cfg['R_GATE']
        MAX_R_DIFF = cfg['MAX_R_DIFF']

        if direction == 'forward':
            ref_point = curr['tail']
            ref_vec = curr['vec']
        else: 
            ref_point = curr['head']
            ref_vec = curr['vec'] 
            
        best_match = None
        best_cost = 999.0
        ref_r = np.linalg.norm(ref_point[:2])
        use_theta = ref_r > R_GATE
        ref_theta = np.arctan2(ref_point[1], ref_point[0]) if use_theta else 0.0

        for cand in pool:
            if cand['used']: continue
            if direction == 'forward':
                pt_norm, pt_flip = cand['head'], cand['tail']
            else:
                pt_norm, pt_flip = cand['tail'], cand['head']
            
            d_norm = np.linalg.norm(ref_point - pt_norm)
            d_flip = np.linalg.norm(ref_point - pt_flip)
            is_flip = False
            
            # 检查是否需要翻转 (Flip)
            if d_flip < d_norm - FLIP_MARGIN:
                is_flip = True
                d = d_flip
                chosen_pt = pt_flip
                chosen_vec = -cand['vec']
            else:
                d = d_norm
                chosen_pt = pt_norm
                chosen_vec = cand['vec']

            # 过滤条件
            if d > MAX_DIST: continue
            if np.dot(ref_vec, chosen_vec) < MIN_COS: continue 
            if abs(ref_point[2] - chosen_pt[2]) > MAX_Z_DIFF: continue
            cand_r = np.linalg.norm(chosen_pt[:2])
            if abs(ref_r - cand_r) > MAX_R_DIFF: continue

            # 角度代价
            theta_cost = 0.0
            if use_theta and cand_r > R_GATE:
                cand_theta = np.arctan2(chosen_pt[1], chosen_pt[0])
                diff = abs(ref_theta - cand_theta)
                diff = min(diff, 2*np.pi - diff)
                if diff > MAX_THETA_DIFF: continue
                theta_cost = diff

            cost = d + 5.0 * (1.0 - np.dot(ref_vec, chosen_vec)) + 0.5 * theta_cost
            if cost < best_cost:
                best_cost = cost
                best_match = (cand, is_flip)
        return best_match

    def _rescue_under_count(self, original_segments):
        """
        救援模式：当识别出的 Strands 太少时，尝试对原始片段进行更激进的切分并重新组装。
        """
        self.audit['rescue_triggered'] = True
        n_before = len(self.strands)
        refined_pool = []
        
        # 使用 min_len=3 进行深度切分
        for seg in original_segments:
            refined_pool.extend(self._deep_split_under_count(seg, min_len=3))

        # 使用更严格的连接参数，防止错误合并
        strict_cfg = self.match_cfg.copy()
        strict_cfg.update({
            'MAX_DIST': 4.0,       
            'MIN_COS': 0.65,       
            'MAX_Z_DIFF': 5.0,     
            'MAX_THETA_DIFF': 0.8, 
            'FLIP_MARGIN': 0.8,
            'MAX_R_DIFF': 2.0      
        })

        new_strands = self._refine_topology(pool_source=refined_pool, cfg=strict_cfg)
        n_after = len(new_strands)
        
        # 判定救援是否成功 (找到至少8条，或者比之前多2条)
        is_success = (n_after >= 8) or (n_after >= n_before + 2)
        if is_success:
            self.strands = new_strands
            self.audit['rescue_success'] = True
            if len(refined_pool) > len(original_segments):
                # 记录额外的切分
                # 注意：这里只是近似计数，因为 audit 是全局累加的
                pass 
        else:
            # 救援失败，保留原样
            pass

    def _fix_parity(self):
        """
        奇偶性修复：β桶必须有偶数条 Strands。
        策略：尝试拆分最长的 Strand，或者丢弃最差的 Strand。
        """
        n = len(self.strands)
        if n % 2 == 0: return

        # 策略 A: 数量很少时 (<=8)，优先尝试几何拆分
        if n <= 8:
            lens = [len(s['coords']) for s in self.strands]
            idx = int(np.argmax(lens))
            coords = self.strands[idx]['coords']
            min_len = 3 

            if len(coords) >= 2 * min_len:
                # 简单的几何拆分逻辑
                split_idx = len(coords) // 2
                # (简化版：省略了复杂的投影寻找最佳点的逻辑，直接取中点，通常足够)
                
                s1 = {'coords': coords[:split_idx], 'vector': self._get_strand_vector(coords[:split_idx])}
                s2 = {'coords': coords[split_idx:], 'vector': self._get_strand_vector(coords[split_idx:])}
                self.strands.pop(idx)
                self.strands.insert(idx, s2)
                self.strands.insert(idx, s1)
                self.audit['parity_method'] = f'GeoSplit_SmallN_{idx}'
                return
            
            self.audit['parity_method'] = 'KeepOdd_SmallN'
            return

        # 策略 B: 数量较多时，评估线性度和长度
        scores = []
        for i, s in enumerate(self.strands):
            c = s['coords']
            v_start = c[-1] - c[0]
            len_direct = np.linalg.norm(v_start)
            len_arc = np.sum([np.linalg.norm(c[k] - c[k-1]) for k in range(1, len(c))])
            linearity = len_direct / (len_arc + 1e-6)
            scores.append({'idx': i, 'linearity': linearity, 'len': len(c)})
        
        if not scores: return
        scores.sort(key=lambda x: x['linearity'])
        cand_split = scores[0] # 线性度最差的（最弯曲，可能是两个连在一起）
        cand_drop = sorted(scores, key=lambda x: x['len'])[0] # 最短的
        
        if cand_split['linearity'] < 0.85 and cand_split['len'] > 8:
            idx = cand_split['idx']
            coords = self.strands[idx]['coords']
            mid = len(coords) // 2
            s1 = {'coords': coords[:mid], 'vector': self._get_strand_vector(coords[:mid])}
            s2 = {'coords': coords[mid:], 'vector': self._get_strand_vector(coords[mid:])}
            self.strands.pop(idx)
            self.strands.insert(idx, s2)
            self.strands.insert(idx, s1)
            self.audit['parity_method'] = f'Split_Strand_{idx}'
        else:
            idx = cand_drop['idx']
            self.strands.pop(idx)
            self.audit['parity_method'] = f'Drop_Strand_{idx}'

    def _decompose_complex_strands(self):
        """检查并拆分过于弯曲（非线性）的 Strands。"""
        max_passes = 3
        min_len = 4
        for _ in range(max_passes):
            new_strands = []
            any_split = False
            for s in self.strands:
                coords = s['coords']
                if len(coords) < min_len * 2: 
                    new_strands.append(s)
                    continue
                
                # 计算线性度
                v_start = coords[-1] - coords[0]
                len_direct = np.linalg.norm(v_start)
                len_path = np.sum([np.linalg.norm(coords[k] - coords[k-1]) for k in range(1, len(coords))])
                linearity = len_direct / (len_path + 1e-6)
                
                if linearity < 0.75:
                    # 尝试拆分
                    split_idx = len(coords) // 2 # 简化拆分点
                    s1 = coords[:split_idx]
                    s2 = coords[split_idx:]
                    if len(s1) >= min_len and len(s2) >= min_len:
                        new_strands.append({'coords': s1, 'vector': self._get_strand_vector(s1)})
                        new_strands.append({'coords': s2, 'vector': self._get_strand_vector(s2)})
                        any_split = True
                        self.audit['n_linear_splits'] += 1
                    else: new_strands.append(s)
                else: new_strands.append(s)
            self.strands = new_strands
            if not any_split: break

    # --- 辅助方法 ---

    def _get_strand_vector(self, coords):
        if len(coords) < 3: return np.array([0,0,1])
        c = coords - np.mean(coords, axis=0)
        # 使用 SVD 计算主方向
        uu, ss, vh = np.linalg.svd(c)
        vec = vh[0]
        # 确保向量指向末端
        if np.dot(vec, coords[-1]-coords[0]) < 0: vec = -vec
        return vec

    def _endpoint_anchor(self, seg, which='head', w=4):
        w = min(w, len(seg))
        pts = seg[:w] if which == 'head' else seg[-w:]
        if len(pts) == 0: return seg[0]
        
        # 简单的加权平均，过滤掉太靠近中心的点（如果需要）
        # 这里简化为直接平均
        return np.mean(pts, axis=0)

    def _flip_segment(self, cand_obj):
        cand_obj['coords'] = cand_obj['coords'][::-1]
        cand_obj['head'], cand_obj['tail'] = cand_obj['tail'], cand_obj['head']
        cand_obj['vec'] = -cand_obj['vec']
        self.audit['n_flips'] += 1

    # --- Rescue Split Utilities (私有辅助，确保 Topology 自包含) ---

    def _deep_split_under_count(self, seg, min_len=4):
        # 组合多种拆分策略
        parts = self._scan_and_split(seg, cos_threshold=0.2, min_seg_len=min_len)
        out = []
        for p in parts:
            for p2 in self._split_by_z_turns(p, min_len=min_len):
                for p3 in self._split_by_theta_span(p2, min_len=min_len):
                    out.append(p3)
        return [x for x in out if len(x) >= 3]

    def _scan_and_split(self, seg, cos_threshold=-0.2, min_seg_len=4):
        # 与 Cleaner 中的逻辑类似，但为了解耦这里保留一份副本
        if len(seg) < min_seg_len * 2: return [seg]
        seg_smooth = seg.copy()
        # 简单平滑
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
            return self._scan_and_split(seg[:best_idx], cos_threshold, min_seg_len) + \
                   self._scan_and_split(seg[best_idx:], cos_threshold, min_seg_len)
        return [seg]

    def _split_by_z_turns(self, seg, min_len=4, dz_eps=0.2):
        # 基于 Z 轴方向变化的拆分 (检测发卡结构)
        if len(seg) < 2 * min_len: return [seg]
        z = seg[:, 2].copy()
        # 平滑 Z
        z_s = z.copy()
        z_s[1:-1] = (z[:-2] + z[1:-1] + z[2:]) / 3.0
        dz = np.diff(z_s)
        dz[np.abs(dz) < dz_eps] = 0.0
        s = np.sign(dz)
        
        # 消除单点抖动
        for i in range(1, len(s)):
            if s[i] == 0: s[i] = s[i-1]
        
        changes = np.where(s[:-1] * s[1:] < 0)[0] + 1
        cuts = [c for c in changes if c >= min_len and (len(seg) - c) >= min_len]
        
        if not cuts: return [seg]
        out, start = [], 0
        for c in cuts:
            out.append(seg[start:c])
            start = c
        out.append(seg[start:])
        return [x for x in out if len(x) >= min_len]

    def _split_by_theta_span(self, seg, min_len=4, r_min=2.0, span_thr=1.3):
        # 基于角度跨度的拆分 (防止一个片段绕桶一圈)
        if len(seg) < 2 * min_len: return [seg]
        r = np.linalg.norm(seg[:, :2], axis=1)
        valid = r > r_min
        if np.sum(valid) < 2 * min_len: return [seg]
        
        theta = np.arctan2(seg[valid, 1], seg[valid, 0])
        theta = np.unwrap(theta)
        span = float(theta.max() - theta.min())
        
        if span < span_thr: return [seg]
        
        # 在角度变化最大的地方切开
        dtheta = np.abs(np.diff(theta))
        k = int(np.argmax(dtheta))
        valid_idx = np.where(valid)[0]
        cut = int(valid_idx[k] + 1)
        
        if cut < min_len or (len(seg) - cut) < min_len: return [seg]
        return [seg[:cut], seg[cut:]]
