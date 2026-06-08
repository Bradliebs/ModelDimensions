# ZCA whitening verification

- Fit samples: **20,000**, eval samples: **20,000**

|  | Raw | Whitened |
| --- | ---: | ---: |
| Effective dim fraction | 0.630 | **0.955** |
| Mean |cosine| | 0.050 | **0.040** |
| Mean norm | 1.000 | 19.668 |

Effective-dim gate (>= 0.95): **PASS**
Pairwise-cosine gate (<= 0.05): **PASS**
Overall: **PASS**

## Notes
- fit and eval samples share size; consider supplying a held-out eval slice to avoid optimistic measurements.
