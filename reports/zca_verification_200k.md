# ZCA whitening verification

- Fit samples: **100,000**, eval samples: **100,000**

|  | Raw | Whitened |
| --- | ---: | ---: |
| Effective dim fraction | 0.636 | **0.985** |
| Mean |cosine| | 0.050 | **0.040** |
| Mean norm | 1.000 | 19.522 |

Effective-dim gate (>= 0.95): **PASS**
Pairwise-cosine gate (<= 0.05): **PASS**
Overall: **PASS**

## Notes
- fit and eval samples share size; consider supplying a held-out eval slice to avoid optimistic measurements.
