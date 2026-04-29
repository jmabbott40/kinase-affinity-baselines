# Aminergic GPCR Data Audit Report

**Date:** 2026-04-29

**Decision:** `OPTION_A`

**Decision message:** 83% pass threshold. Proceed with binding-only data.

## Summary statistics

- Total targets queried: 36
- Targets passing threshold (>=500 binding records): 30
- Pass fraction: 83.3%

## Per-family breakdown

| family      |   n_targets |   n_passing |   mean_records |   median_records |
|:------------|------------:|------------:|---------------:|-----------------:|
| adrenergic  |           9 |           8 |           1235 |             1255 |
| dopamine    |           5 |           4 |           5286 |             3850 |
| histamine   |           4 |           4 |           2289 |             1854 |
| muscarinic  |           5 |           5 |           1987 |             2110 |
| serotonin   |          12 |           9 |           2682 |             1732 |
| trace_amine |           1 |           0 |             47 |               47 |

## Per-target details

| target_chembl_id   |   n_binding_records | n_unique_compounds   | type_breakdown                        | gene_symbol   | family      | passes_threshold   |
|:-------------------|--------------------:|:---------------------|:--------------------------------------|:--------------|:------------|:-------------------|
| CHEMBL2056         |                1923 |                      | {'IC50': 162, 'Ki': 1724, 'Kd': 37}   | DRD1          | dopamine    | True               |
| CHEMBL217          |               12141 |                      | {'IC50': 637, 'Ki': 11375, 'Kd': 129} | DRD2          | dopamine    | True               |
| CHEMBL234          |                8079 |                      | {'IC50': 310, 'Ki': 7750, 'Kd': 19}   | DRD3          | dopamine    | True               |
| CHEMBL219          |                3850 |                      | {'IC50': 363, 'Ki': 3470, 'Kd': 17}   | DRD4          | dopamine    | True               |
| CHEMBL1850         |                 437 |                      | {'IC50': 13, 'Ki': 420, 'Kd': 4}      | DRD5          | dopamine    | False              |
| CHEMBL214          |                6471 |                      | {'IC50': 451, 'Ki': 5953, 'Kd': 67}   | HTR1A         | serotonin   | True               |
| CHEMBL1898         |                1173 |                      | {'IC50': 328, 'Ki': 844, 'Kd': 1}     | HTR1B         | serotonin   | True               |
| CHEMBL1983         |                1462 |                      | {'IC50': 404, 'Ki': 1056, 'Kd': 2}    | HTR1D         | serotonin   | True               |
| CHEMBL2182         |                  86 |                      | {'IC50': 4, 'Ki': 82, 'Kd': 0}        | HTR1E         | serotonin   | False              |
| CHEMBL1805         |                 127 |                      | {'IC50': 3, 'Ki': 124, 'Kd': 0}       | HTR1F         | serotonin   | False              |
| CHEMBL224          |                7201 |                      | {'IC50': 1033, 'Ki': 6162, 'Kd': 6}   | HTR2A         | serotonin   | True               |
| CHEMBL1833         |                2002 |                      | {'IC50': 217, 'Ki': 1784, 'Kd': 1}    | HTR2B         | serotonin   | True               |
| CHEMBL225          |                3973 |                      | {'IC50': 865, 'Ki': 3106, 'Kd': 2}    | HTR2C         | serotonin   | True               |
| CHEMBL1875         |                 566 |                      | {'IC50': 67, 'Ki': 497, 'Kd': 2}      | HTR4          | serotonin   | True               |
| CHEMBL3426         |                 422 |                      | {'IC50': 32, 'Ki': 390, 'Kd': 0}      | HTR5A         | serotonin   | False              |
| CHEMBL3371         |                4919 |                      | {'IC50': 386, 'Ki': 4532, 'Kd': 1}    | HTR6          | serotonin   | True               |
| CHEMBL3155         |                3785 |                      | {'IC50': 548, 'Ki': 3228, 'Kd': 9}    | HTR7          | serotonin   | True               |
| CHEMBL229          |                1927 |                      | {'IC50': 184, 'Ki': 1732, 'Kd': 11}   | ADRA1A        | adrenergic  | True               |
| CHEMBL232          |                1556 |                      | {'IC50': 80, 'Ki': 1476, 'Kd': 0}     | ADRA1B        | adrenergic  | True               |
| CHEMBL223          |                1557 |                      | {'IC50': 172, 'Ki': 1385, 'Kd': 0}    | ADRA1D        | adrenergic  | True               |
| CHEMBL1867         |                1255 |                      | {'IC50': 227, 'Ki': 1018, 'Kd': 10}   | ADRA2A        | adrenergic  | True               |
| CHEMBL1942         |                 845 |                      | {'IC50': 142, 'Ki': 702, 'Kd': 1}     | ADRA2B        | adrenergic  | True               |
| CHEMBL1916         |                 992 |                      | {'IC50': 129, 'Ki': 862, 'Kd': 1}     | ADRA2C        | adrenergic  | True               |
| CHEMBL213          |                 994 |                      | {'IC50': 535, 'Ki': 386, 'Kd': 73}    | ADRB1         | adrenergic  | True               |
| CHEMBL210          |                1540 |                      | {'IC50': 559, 'Ki': 789, 'Kd': 192}   | ADRB2         | adrenergic  | True               |
| CHEMBL246          |                 453 |                      | {'IC50': 79, 'Ki': 357, 'Kd': 17}     | ADRB3         | adrenergic  | False              |
| CHEMBL231          |                1743 |                      | {'IC50': 255, 'Ki': 1352, 'Kd': 136}  | HRH1          | histamine   | True               |
| CHEMBL1941         |                 708 |                      | {'IC50': 108, 'Ki': 528, 'Kd': 72}    | HRH2          | histamine   | True               |
| CHEMBL264          |                4738 |                      | {'IC50': 308, 'Ki': 4343, 'Kd': 87}   | HRH3          | histamine   | True               |
| CHEMBL3759         |                1966 |                      | {'IC50': 139, 'Ki': 1752, 'Kd': 75}   | HRH4          | histamine   | True               |
| CHEMBL216          |                2110 |                      | {'IC50': 533, 'Ki': 1518, 'Kd': 59}   | CHRM1         | muscarinic  | True               |
| CHEMBL211          |                2314 |                      | {'IC50': 384, 'Ki': 1869, 'Kd': 61}   | CHRM2         | muscarinic  | True               |
| CHEMBL245          |                3034 |                      | {'IC50': 755, 'Ki': 2268, 'Kd': 11}   | CHRM3         | muscarinic  | True               |
| CHEMBL1821         |                1502 |                      | {'IC50': 226, 'Ki': 1270, 'Kd': 6}    | CHRM4         | muscarinic  | True               |
| CHEMBL2035         |                 975 |                      | {'IC50': 195, 'Ki': 776, 'Kd': 4}     | CHRM5         | muscarinic  | True               |
| CHEMBL5857         |                  47 |                      | {'IC50': 0, 'Ki': 47, 'Kd': 0}        | TAAR1         | trace_amine | False              |