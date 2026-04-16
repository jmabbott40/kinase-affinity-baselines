# Dataset Card: Kinase Inhibitor Bioactivity (v1)

## Overview

Curated protein kinase binding affinity dataset from ChEMBL for benchmarking ML models.

| Property | Value |
|----------|-------|
| Source | ChEMBL 36 |
| Target family | Protein kinases (Homo sapiens) |
| Activity types | IC50, Ki, Kd |
| Response variable | pActivity (−log₁₀ M) |
| Dataset version | v1 |
| Creation date | March 2026 |

## Inclusion Criteria

- **Target type**: SINGLE PROTEIN targets in the protein kinase family
- **Organism**: Homo sapiens only
- **Activity types**: IC50, Ki, Kd measurements
- **Standard relation**: Exact values only (`=`), no inequalities (`>`, `<`)
- **Units**: Nanomolar (nM), with pChEMBL value present
- **Assay confidence**: Score ≥ 7 (direct single protein target assignment)

## Molecule Standardization

Applied using RDKit:
1. Salt removal (keep largest fragment)
2. Charge neutralization
3. Canonical SMILES generation
4. Molecular weight filter: 100–900 Da
5. Maximum 100 heavy atoms

**Molecules removed**: TBD (to be filled after pipeline runs)

## Duplicate Handling

For identical (canonical_smiles, target_chembl_id, activity_type) groups:
- Aggregation: **median** pActivity value
- Noise flag: if ≥3 measurements and std > 1.0 pActivity units, compound is flagged as "noisy"
- All measurements contribute to the median regardless of noise flag

## Quality Filters

- pActivity range: [3.0, 12.0] (removes values below 1 mM or above 1 pM)
- Classification threshold: pActivity ≥ 6.0 (IC50 ≤ 1 μM) = **active**

## Splitting Strategies

### Random Split
- 80% train / 10% validation / 10% test
- Stratified by target to maintain target representation in all splits
- Seed: 42

### Scaffold Split
- Murcko generic scaffolds (ring systems only)
- Entire scaffold groups assigned to one split (no scaffold leakage)
- Greedy assignment sorted by scaffold group size
- Seed: 42

### Target Split
- Entire kinase subfamilies held out for testing
- Tests generalization to unseen protein targets
- Seed: 42

## Known Limitations

- **Activity type mixing**: IC50, Ki, and Kd are not directly comparable. IC50 values depend on assay conditions (substrate concentration, etc.). Combining them adds noise.
- **Assay heterogeneity**: Measurements from different labs and assay formats are combined. Inter-lab variability is a known source of noise.
- **Stereochemistry**: Some compounds may have unresolved stereochemistry, which affects binding but is lost in 2D representations.
- **Temporal bias**: Older ChEMBL entries may have lower data quality.

## Citation

If using this dataset, please cite:
- ChEMBL: Mendez et al. (2019) "ChEMBL: towards direct deposition of bioassay data" Nucleic Acids Research, 47(D1), D930-D940.
