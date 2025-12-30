import os
import sys
import warnings
import shutil
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# Suppress non-critical Biopython warnings (e.g. discontinuous chains)
warnings.simplefilter('ignore', BiopythonWarning)

class ProteinParser:
    """
    Robust loader for PDB/MMCIF files with integrated DSSP execution.
    Handles file format detection and provides clean access to structure/sequence.
    """

    def __init__(self, file_path, model_id=0):
        self.file_path = file_path
        self.model_id = model_id
        self.structure = None
        self.model = None
        self.dssp = None
        self._load_structure()

    def _load_structure(self):
        """Auto-detects format (.pdb/.cif) and loads structure."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Structure file not found: {self.file_path}")

        # Logic: Detect parser based on extension
        ext = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if ext in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
                self.structure = parser.get_structure('struct', self.file_path)
            else:
                # Default to PDB parser for .pdb, .ent, or no extension
                parser = PDBParser(QUIET=True)
                self.structure = parser.get_structure('struct', self.file_path)
            
            self.model = self.structure[self.model_id]
            
        except Exception as e:
            raise ValueError(f"Failed to parse structure {self.file_path}: {str(e)}")

    def run_dssp(self, dssp_bin=None):
        """
        Executes DSSP with verbose debugging and explicit binary path.
        """
        if self.dssp is not None:
            return self.dssp

        # --- FIX: Explicitly find binary if not provided ---
        if dssp_bin is None:
            # 优先尝试 mkdssp (Conda 环境通常叫这个)
            if shutil.which("mkdssp"):
                dssp_bin = shutil.which("mkdssp")
            elif shutil.which("dssp"):
                dssp_bin = shutil.which("dssp")
            # 如果上面都找不到，尝试硬编码 (请根据你的 'which mkdssp' 结果修改这里!)
            # else:
            #     dssp_bin = "/home/zero/miniconda3/envs/evofast/bin/mkdssp"
        
        if not dssp_bin or not os.path.exists(dssp_bin):
            raise RuntimeError(f"DSSP binary not found. Looked for 'mkdssp'/'dssp'. PATH={os.environ.get('PATH')}")

        # --- FIX: Use absolute path for PDB file ---
        abs_path = os.path.abspath(self.file_path)

        try:
            # Biopython 1.78+ DSSP class
            self.dssp = DSSP(self.model, abs_path, dssp=dssp_bin)
            return self.dssp
        except Exception as e:
            # 打印详细错误到控制台，方便调试
            print(f"!!! DSSP CRASHED on {abs_path} using binary {dssp_bin} !!!")
            print(f"Error details: {str(e)}")
            raise RuntimeError(f"DSSP failed: {str(e)}")
        
    def get_chain(self, chain_id):
        """Safe chain retrieval."""
        if chain_id in self.model:
            return self.model[chain_id]
        return None

    def get_sequence(self, chain_id):
        """
        Extracts the 1-letter amino acid sequence for a chain.
        Useful for checking sequence identity later.
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return ""
        
        seq = []
        for residue in chain:
            # Only count standard amino acids (ignore waters, ligands)
            if is_aa(residue, standard=True):
                seq.append(seq1(residue.get_resname()))
        return "".join(seq)

    def get_ca_coords(self, chain_id):
        """
        Returns list of CA coordinates for a chain.
        Format: [(res_num, [x, y, z]), ...]
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return []
            
        coords = []
        for res in chain:
            if is_aa(res) and 'CA' in res:
                coords.append((res.id[1], res['CA'].get_coord()))
        return coords

# --- Quick Test ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdb_file>")
        sys.exit(1)
        
    pdb_file = sys.argv[1]
    try:
        loader = ProteinParser(pdb_file)
        print(f"Loaded: {os.path.basename(pdb_file)}")
        
        # Try running DSSP
        # Note: You need 'mkdssp' installed for this to work
        # on Ubuntu: sudo apt-get install dssp
        # on Conda: conda install -c salilab dssp
        try:
            dssp_data = loader.run_dssp()
            print(f"DSSP Run Success. Total residues processed: {len(dssp_data)}")
        except RuntimeError as e:
            print(f"DSSP Skipped: {e}")

        # Get Sequence of Chain A
        seq = loader.get_sequence('A')
        print(f"Chain A Sequence ({len(seq)} residues): {seq[:20]}...")
        
    except Exception as e:
        print(f"Error: {e}")
