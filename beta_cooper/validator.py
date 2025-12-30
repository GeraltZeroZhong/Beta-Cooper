import numpy as np
import warnings
import os
import shutil
import subprocess
import tempfile
from Bio.PDB import PDBParser
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet

class BarrelValidator:
    """
    BarrelValidator V25 (OPM Optimized)
    
    Updates:
    - Optimized for Natural Proteins (OPM database).
    - Disabled strict penalties for thickness and circularity.
    - Lowered validation threshold.
    """

    def __init__(self, pdb_file, chain_id='A'):
        self.pdb_file = pdb_file
        self.target_chain_id = chain_id 
        self.parser = PDBParser(QUIET=True)
        try:
            with open(pdb_file, 'r'): pass
            self.file_exists = True
        except:
            self.file_exists = False

    def validate(self):
        metrics = self._get_empty_metrics()
        
        if not self.file_exists:
             return self._fail("FAIL_INPUT", "File not found", metrics)

        # 1. Robust Extraction
        segments, all_coords = self._extract_segments_v24()
        
        if segments is None:
            return self._fail("FAIL_DSSP_EXEC", "DSSP execution failed", metrics)

        metrics["n_segments"] = len(segments)
        metrics["n_atoms_total"] = len(all_coords)
        
        if len(all_coords) < 30:
            return self._fail("FAIL_LOW_BETA", f"Insufficient beta atoms ({len(all_coords)}<30)", metrics)

        # 2. PCA & Geometry Checks
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
        
        if axis_dominance_ratio < 1.15:
            best_plane, best_cv = self._find_best_plane_fallback(aligned, vars, axis_dominance_ratio)
            if best_plane is None:
                return self._fail("FAIL_AMBIGUOUS", "Shape ambiguous (Fallback failed)", metrics)
            plane_idx = best_plane
            z_idx = list(set([0, 1, 2]) - set(plane_idx))[0]
            plane_source = "Fallback_Search"
        
        metrics["plane_source"] = plane_source

        # 3. Inlier Selection
        pts_2d = aligned[:, plane_idx]
        
        try:
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
            
            # 4. Detailed Metrics
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
            
            metrics.update({
                "ellipticity": ellipticity,
                "cov_eigen_ratio": cov_eigen_ratio,
                "ring_cv": ring_cv,
                "rad_kurtosis": rad_kurtosis,
                "local_thickness": local_thickness,
                "angular_gap": angular_gap,
                "n_angular_bins": n_bins
            })

        except Exception as e:
             return self._fail("FAIL_MATH", f"Math Error: {str(e)}", metrics)

        # 5. Stability & Penalty
        try:
            sort_idx = np.argsort(z_core)
            r_sorted = radii[sort_idx]
            n_chunks = max(2, min(4, len(z_core) // 15))
            chunks = np.array_split(r_sorted, n_chunks)
            chunk_means = [np.mean(c) for c in chunks if len(c) > 0]
            z_profile_cv = np.std(chunk_means)/(np.mean(chunk_means)+1e-6) if len(chunk_means)>1 else 0.0
            metrics["z_profile_cv"] = z_profile_cv
        except:
            metrics["z_profile_cv"] = 1.0

        if ellipticity > 1.8: metrics["pen_ellipticity"] = (ellipticity - 1.8) * 0.6
        if cov_eigen_ratio > 50: metrics["pen_stability"] = min(0.3, 0.05 * np.log10(cov_eigen_ratio / 50))
        
        # [MODIFIED for OPM] Disabled Ring CV Penalty
        # cv_thr = 0.20 if is_small_sample else 0.25
        # cv_mult = 3.5 if is_small_sample else 2.5
        # if ring_cv > cv_thr: metrics["pen_ringcv"] = (ring_cv - cv_thr) * cv_mult
        
        # [MODIFIED for OPM] Disabled Thickness Penalty
        # thick_thr = 0.25 if is_small_sample else 0.30
        # thick_mult = 3.0 if is_small_sample else 2.5
        # if local_thickness > thick_thr: metrics["pen_thickness"] = (local_thickness - thick_thr) * thick_mult
        
        if not is_small_sample:
            if local_thickness > 0.35 and rad_kurtosis < -1.0: metrics["pen_kurtosis"] = 0.5 
            elif rad_kurtosis < -1.0: metrics["pen_kurtosis"] = 0.1
        
        if angular_gap > 60: metrics["pen_gap"] = (angular_gap - 60) / 100.0
        if metrics.get("z_profile_cv", 0) > 0.3: metrics["pen_z"] = 0.3
        
        if plane_source != "PC1_Axis": metrics["pen_fallback"] = 0.15
        if metrics["n_segments"] < 3: metrics["pen_segments"] = 0.25

        total_penalty = sum(v for k, v in metrics.items() if k.startswith("pen_"))
        score = max(0.0, 1.0 - total_penalty)
        
        # Lowered threshold 
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
            "metrics": metrics,
            "debug_segments": segments,    # EXPOSE DATA
            "debug_coords": all_coords     # EXPOSE DATA
        }

    def _extract_segments_v24(self):
        dssp_bin = shutil.which("mkdssp") or shutil.which("dssp") or "/usr/bin/mkdssp"
        if not os.path.exists(dssp_bin): return None, None
        pdb_abspath = os.path.abspath(self.pdb_file)
        
        # 1. Sanitize
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_in:
            sanitized_path = tmp_in.name
            try:
                self._sanitize_opm(pdb_abspath, tmp_in)
            except Exception:
                tmp_in.close()
                if os.path.exists(sanitized_path): os.remove(sanitized_path)
                return None, None
        
        # 2. Run DSSP (Legacy)
        with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp_out:
            output_file_path = tmp_out.name
        try:
            cmd = [dssp_bin, sanitized_path, output_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0: return None, None
            dssp_data = self._parse_dssp_raw(output_file_path)
        except Exception: return None, None
        finally:
            if os.path.exists(output_file_path): os.remove(output_file_path)

        # 3. Reload & Map
        clean_parser = PDBParser(QUIET=True)
        try:
            clean_struct = clean_parser.get_structure('clean', sanitized_path)
            clean_model = clean_struct[0]
        except Exception:
            if os.path.exists(sanitized_path): os.remove(sanitized_path)
            return None, None
        if os.path.exists(sanitized_path): os.remove(sanitized_path)

        target = self.target_chain_id if self.target_chain_id and self.target_chain_id.strip() else 'A'
        if target in clean_model: chain_obj = clean_model[target]
        elif 'A' in clean_model: chain_obj = clean_model['A']
        else: chain_obj = list(clean_model)[0]
        
        segments = []
        current_segment = []
        last_resseq = -999
        JUMP_TOL = 4
        
        for residue in chain_obj:
            if not residue.has_id('CA'): continue
            c_id = chain_obj.id
            r_seq = residue.id[1]
            ss = dssp_data.get((c_id, r_seq))
            if ss is None:
                if c_id == 'A': ss = dssp_data.get((' ', r_seq))
                elif c_id == ' ': ss = dssp_data.get(('A', r_seq))
            
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

    def _parse_dssp_raw(self, dssp_file):
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
        output_handle.write("HEADER    OPM CLEANED PDB\n")
        output_handle.write("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
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
        return {"status": status, "is_valid": False, "confidence": 0.0, "issue": reason, "metrics": metrics, "debug_segments": None, "debug_coords": None}

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