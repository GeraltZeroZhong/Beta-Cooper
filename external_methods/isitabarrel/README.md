# IsItABarrel Structure-Map Baseline

This adapter treats
[SluskyLab/isitabarrel](https://github.com/SluskyLab/isitabarrel) as an
external baseline for Cooper-Beta evaluation. Cooper-Beta labels this variant
`isitabarrel_structure_map` because it runs the IsItABarrel heuristics on
contact maps derived from PDB/CIF structures, not on the original evolutionary
contact maps used in the publication.

The official IsItABarrel script is not vendored here. It is licensed under
AGPL-3.0, while Cooper-Beta is MIT-licensed, so this directory only contains
the invocation and result-parsing layer.

## Expected Inputs For The Upstream Runner

The upstream script expects:

- a protein-id list, one id per line or as the first tab-separated column;
- a directory containing one contact-map pickle per id, named `<id>.pkl`;
- an official `isitabarrel.py` checkout, provided with `--script` or the
  `ISITABARREL_SCRIPT` environment variable.

The upstream program writes `results.tsv` in its working directory. The adapter
parses that file and normalizes each row into:

- `baseline`: `isitabarrel_structure_map`
- `sample_id`: upstream `MAP_NAME`
- `result`: `BARREL` when the selected score is greater than zero, otherwise
  `NON_BARREL`
- `score`: selected decision score, defaulting to `CC2_TO_H4`, which the
  upstream comments recommend for reproducing the publication results

## Structure Map Workflow

When you already have PDB/CIF/mmCIF structures, generate structure-derived
contact maps directly instead of running a FASTA-to-contact-map prediction
pipeline:

```bash
python external_methods/isitabarrel/structure_map.py \
  path/to/structures \
  --out-dir eval_outputs/isitabarrel_structure_map \
  --script path/to/isitabarrel.py \
  --out eval_outputs/isitabarrel_structure_map.csv
```

The generator writes:

- `maps/<sample_id>.pkl`: NumPy `float32` `L x L` contact maps
- `protid_list.tsv`: sample ids for the upstream script
- `residue_mapping.csv`: matrix index to source residue mapping

The default map uses CA-CA contacts within 8.0 Angstrom, zeros contacts with
sequence distance of two residues or less, and skips chains with fewer than 15
CA residues to avoid an upstream IsItABarrel indexing failure on very short
chains.

## Existing Map Example

```bash
python external_methods/isitabarrel/runner.py \
  path/to/protid_list.tsv \
  path/to/contact_maps \
  --script path/to/isitabarrel.py \
  --out eval_outputs/isitabarrel_structure_map.csv
```

For smoke tests, this repository uses a tiny fake upstream script under
`data/external_methods/isitabarrel_smoke/` so the adapter can be tested without
vendoring or downloading AGPL code.
