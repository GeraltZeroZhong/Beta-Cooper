# Foldseek Global-TMalign Structure-Search Baseline

This adapter treats [Foldseek](https://github.com/steineggerlab/foldseek) as
an external Cooper-Beta evaluation baseline. Cooper-Beta labels this variant
`foldseek_tmalign_structure_search` because it searches query chains against a
reference set of known beta-barrel chains using Foldseek's global TMalign mode.

Foldseek is GPL-3.0, while Cooper-Beta is MIT-licensed, so the upstream binary
and any Foldseek databases are not vendored here. This directory only contains
chain-file generation, invocation, and result normalization.

## Upstream Sources

- Foldseek repository: <https://github.com/steineggerlab/foldseek>
- Foldseek paper: van Kempen M. et al. Nature Biotechnology. 2024.

## Decision Rule

The main baseline uses `--alignment-type 1`, Foldseek's global TMalign mode.
This is intentionally different from Foldseek's default local 3Di+AA alignment:
the Cooper-Beta task asks whether a whole chain has a beta-barrel-like fold, not
whether it contains a locally similar motif.

The normalized decision defaults are:

- `score`: `min(qtmscore, ttmscore)` for the best eligible hit
- `result`: `BARREL` when `score >= 0.50`, query coverage is at least `0.60`,
  and target coverage is at least `0.60`; otherwise `NON_BARREL`
- `decision_rule`: recorded in each row so thresholds can be recalibrated

The normalized CSV keeps the best target id, coverage, `qtmscore`, `ttmscore`,
`alntmscore`, Foldseek score fields, and hit counts.

## Reference Database

For repeated evaluation, build a custom reference database once from curated
canonical beta-barrel chain structures:

```bash
foldseek createdb path/to/reference_barrel_chains ref_barrel_db
foldseek createindex ref_barrel_db tmp_index
```

The reference set and a family-group manifest are mandatory for dataset
evaluation. The manifest must assign every query and reference chain to a
homology group derived independently of the evaluation labels. All reference
targets from the query's group and PDB id are excluded. In addition, the
evaluator hashes the exact bytes of every generated single-chain PDB artifact
and excludes every reference whose normalized-chain SHA-256 equals the query's,
even if the source was renamed or assigned an incorrect group. Hash coverage is
validated exactly and fails closed when an artifact cannot be compared
reliably. PDB-only exclusion is not sufficient because homologous structures
can occur under different PDB identifiers.

## Structure-to-Search Workflow

When starting from PDB/CIF/mmCIF query structures, generate one single-chain PDB
record per analyzable chain and run the baseline:

```bash
python external_methods/foldseek/structure_search.py \
  path/to/query_structures \
  --out-dir eval_outputs/foldseek_tmalign_structure_search \
  --target-db path/to/ref_barrel_db \
  --out eval_outputs/foldseek_tmalign_structure_search.csv
```

You can also point the adapter at reference structures directly and let it run
`foldseek createdb` in the working directory:

```bash
python external_methods/foldseek/structure_search.py \
  path/to/query_structures \
  --out-dir eval_outputs/foldseek_tmalign_structure_search \
  --reference-structures path/to/reference_barrel_chains \
  --create-index \
  --out eval_outputs/foldseek_tmalign_structure_search.csv
```

The generator writes:

- `query_chains/chains/<sample_id>.pdb`: one exported chain per sample
- `query_chains/chain_manifest.csv`: source file and chain metadata
- `query_chains/residue_mapping.csv`: chain-file residue index to source residue
- `foldseek_work/foldseek_hits.tsv`: raw Foldseek TSV before normalization

For smoke tests, this repository uses a tiny fake Foldseek runner under
`data/external_methods/foldseek_smoke/` so the adapter can be tested without
installing or vendoring GPL code.

## Cooper-Beta Dataset Evaluation

The dataset evaluator defaults to directory-labelled file metrics: each input
structure is one observation and is predicted positive when any of its chains
is predicted as a barrel. Chain metrics are available only with paired,
pre-specified target-chain manifests for both positive and negative splits.
Each target manifest must have exactly the columns `relative_path,author_chain_id` and
exactly one row for every structure file. Other chains in the same structure
remain unlabeled and never enter chain-level denominators.

```bash
python external_methods/foldseek/evaluate_dataset.py \
  --positive-dir path/to/positive_structures \
  --negative-dir path/to/negative_structures \
  --reference-dir path/to/reference_barrel_chains \
  --group-manifest path/to/foldseek_family_groups.csv \
  --out-dir eval_outputs \
  --foldseek path/to/foldseek \
  --metric-level both \
  --positive-target-manifest path/to/positive_target_chains.csv \
  --negative-target-manifest path/to/negative_target_chains.csv \
  --tag run-1.0.0
```

There is no implicit reference set. The group manifest has the strict columns
`split,relative_path,author_chain_id,group_id`, must cover every generated chain in the
`positive`, `negative`, and `reference` inputs, and must be frozen alongside
the benchmark. `author_chain_id` is the exact non-blank author-chain identifier. Rows
with blank author-chain identifiers are rejected because the evaluator requires the
exact identifier. The
evaluator excludes every same-family and same-PDB reference
target, plus every exact normalized-chain content match, before choosing the
best hit. The manifest records the hash policy, query and reference inventory
hashes, per-query exclusion categories, and total candidate/observed exclusion
counts; all normalized-chain hashes are revalidated before completion. Every
invocation creates a fresh UTC timestamped run directory under `--out-dir`;
existing runs are never reused or overwritten. It writes:

- `chain_predictions.csv`: all predictions; `is_target_author_chain` identifies
  the manifest-selected author chain, and only those rows have labels
- `file_results.csv`: directory-labelled any-chain file observations
- `target_chain_results.csv`: present only for `chain`/`both` metrics
- `summary.csv` and `summary.md`: the single predeclared metric scope
- `evaluation_manifest.json`: running/failed/complete provenance, including
  normalized parameters, input and executable hashes, sampling semantics, and
  hashes for every run artifact

Manual post-hoc relabeling is intentionally unsupported. Freeze corrected
scientific labels in the input split and target manifests before evaluation.
Missing predictions, invalid results, non-finite scores, unmatched manifest
targets, and structures that produce no eligible chain all fail closed.
