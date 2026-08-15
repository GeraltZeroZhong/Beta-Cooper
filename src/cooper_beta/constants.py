from __future__ import annotations

POLYMER_POSITION_POLICY = "selected-model-mmcif-label-seq-or-pdb-unique-seqres"
DSSP_RESIDUE_COVERAGE_POLICY = (
    "selected-model-declared-polypeptide-ca-residues-with-finite-n-ca-c-o"
)

RESULT_BARREL = "BARREL"
RESULT_NON_BARREL = "NON_BARREL"
RESULT_ERROR = "ERROR"

RESULT_STAGE_PREPARATION = "preparation"
RESULT_STAGE_DECISION = "decision"
RESULT_STAGE_WORKER = "worker"
RESULT_STAGES = frozenset({RESULT_STAGE_PREPARATION, RESULT_STAGE_DECISION, RESULT_STAGE_WORKER})

DEFAULT_RESULT_COLUMNS = (
    "filename",
    "source_path",
    "author_chain_id",
    "result",
    "result_stage",
    "dssp_unassigned_residue_count",
    "strand_count",
    "strand_adjacency_count",
    "cycle_strand_count",
    "cycle_strand_fraction",
    "cycle_rank",
    "reason",
    "error_code",
    "degraded",
)

RULE_MEASUREMENT_COLUMNS = (
    "strand_adjacency_count",
    "cycle_strand_count",
    "cycle_strand_fraction",
    "cycle_rank",
)

DEFAULT_SUMMARY_COLUMNS = (
    "filename",
    "author_chain_id",
    "result",
    "strand_adjacency_count",
    "cycle_strand_count",
    "cycle_strand_fraction",
    "cycle_rank",
    "reason",
)

SUMMARY_COLUMN_WIDTHS = {
    "filename": 20,
    "author_chain_id": 15,
    "result": 14,
    "strand_adjacency_count": 12,
    "cycle_strand_count": 13,
    "cycle_strand_fraction": 15,
    "cycle_rank": 10,
    "reason": 72,
}

NATIVE_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# Tolerance for checking a fraction serialized to text against its integer counts.
SERIALIZED_FRACTION_ABSOLUTE_TOLERANCE = 1.0e-12
