# PRED-TMBB2 Single-Sequence JUCHMME Baseline

This adapter treats PRED-TMBB2 single-sequence topology prediction as an
external Cooper-Beta evaluation baseline. Cooper-Beta labels this variant
`pred_tmbb2_single_juchmme` because it uses the local JUCHMME implementation
and derives a binary decision from the predicted topology.

The upstream JUCHMME package is GPL-3.0, while Cooper-Beta is MIT-licensed, so
the upstream Java code and trained parameter files are not vendored here. This
directory only contains invocation, FASTA generation, and result normalization.

## Upstream Sources

- PRED-TMBB2 web page: <https://hannibal.dib.uth.gr/PRED-TMBB2/>
- JUCHMME source and releases: <https://github.com/pbagos/juchmme>
- PRED-TMBB2 paper: Tsirigos KD, Elofsson A, Bagos PG. Bioinformatics. 2016.

## Expected Local Layout

Download and unpack a JUCHMME release, then provide the release root with
`--juchmme-dir` or the `PRED_TMBB2_JUCHMME_DIR` environment variable. The
adapter expects the standard release paths:

- `bin/`: compiled Java classes
- `models/tmbb2.mdel`
- `tables/A_TMBB2_TRAINED`
- `tables/E_TMBB2_TRAINED`
- `conf/conf.tmbb`

## Decision Rule

JUCHMME emits topology strings and scores, but the full PRED-TMBB2 web-server
discrimination pipeline also included optional signal-peptide and Pfam/OMPdb
features. This adapter therefore uses an explicit topology-derived decision:

- `result`: `BARREL` when the selected topology field has at least three
  predicted `M` runs, otherwise `NON_BARREL`
- `score`: number of predicted `M` runs
- default selected topology: `LP`

The normalized CSV also keeps `logodds`, reliability, algorithm score, sequence
length, and the raw topology string so downstream analyses can use a different
threshold if needed.

Dataset evaluation requires both the reported sequence length and topology
length to match the declared complete polymer sequence exactly. It recomputes
the strand count and binary decision from that topology; truncated, shifted, or
inconsistent upstream records fail the run.

## Existing FASTA Example

```bash
python external_methods/pred_tmbb2/runner.py \
  path/to/sequences.fasta \
  --juchmme-dir path/to/juchmme \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

## Structure-to-Sequence Workflow

When starting from PDB/CIF/mmCIF structures, generate one chain-level FASTA
record per declared protein polymer and run the baseline:

```bash
python external_methods/pred_tmbb2/structure_sequence.py \
  path/to/structures \
  --out-dir eval_outputs/pred_tmbb2_single_juchmme \
  --juchmme-dir path/to/juchmme \
  --out eval_outputs/pred_tmbb2_single_juchmme.csv
```

The generator writes:

- `sequences.fasta`: one FASTA record per chain
- `residue_mapping.csv`: every complete polymer-sequence position, its source
  monomer, exact `author_chain_id`, label-chain and entity identifiers, sequence
  source, and sequence SHA-256
- `juchmme_work/`: upstream working directory

PRED-TMBB2 is a sequence method, so the generator never reconstructs its input
from the subset of residues that happen to have coordinates. PDB input must
contain complete `SEQRES` records. mmCIF input must contain consistent
`_entity_poly`, `_entity_poly_seq`, and `_struct_asym` categories, plus an exact
label-to-author-chain mapping from `_pdbx_poly_seq_scheme` and/or `_atom_site`.
Missing or ambiguous declarations fail closed; there is no observed-CA fallback.

For smoke tests, this repository uses a tiny fake JUCHMME runner under
`data/external_methods/pred_tmbb2_smoke/` so the adapter can be tested without
vendoring or downloading GPL code.

## Cooper-Beta Dataset Evaluation

Dataset evaluation defaults to one directory-labelled observation per
structure file. A file is positive when any generated chain is predicted as a
barrel. This is the only valid metric without target-chain manifests:

```bash
python external_methods/pred_tmbb2/evaluate_dataset.py \
  --positive-dir path/to/positive_structures \
  --negative-dir path/to/negative_structures \
  --juchmme-dir path/to/juchmme \
  --out-dir eval_outputs \
  --metric-level file \
  --tag run-1.0.0
```

For chain metrics, provide both `--positive-target-manifest` and
`--negative-target-manifest` and choose `--metric-level chain` or `both`.
Each CSV must have exactly `relative_path,author_chain_id` columns and exactly one target
chain for every structure file. Partner chains remain unlabeled and do not
enter chain-level metrics. Manual post-hoc exclusions or relabeling are not
supported.

Every invocation creates a fresh UTC timestamped directory below `--out-dir`
containing `chain_predictions.csv` (whose `is_target_author_chain` field marks
the manifest-selected author chain), `file_results.csv`, optional
`target_chain_results.csv`, `summary.csv`, `summary.md`, and
`evaluation_manifest.json`. The manifest is written as `running` immediately,
then atomically transitions to `failed` or `complete`; it records input hashes,
normalized parameters, the Java executable, the complete non-`.git` JUCHMME
checkout inventory, sampling semantics, and all artifact hashes. Missing or
ERROR results, non-finite scores, unmatched targets, and structures without an
eligible predicted chain fail closed.
