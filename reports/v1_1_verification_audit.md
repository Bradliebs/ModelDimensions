# V1.1 verification audit

Cases: 24

## Metrics
| Metric | Events | Total | Observed | Wilson 95% | Rule-of-three upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| coverage | 4 | 24 | 0.167 | 0.067-0.359 |  |
| false_accept_rate | 0 | 13 | 0.000 | 0.000-0.228 | 0.231 |
| over_refusal_rate | 7 | 11 | 0.636 | 0.354-0.848 |  |
| accuracy_among_answered | 4 | 4 | 1.000 | 0.510-1.000 |  |

## Risk / Coverage Curve
| Coverage threshold | Answered | Coverage | False accepts | Error among answered |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 21/24 | 0.875 | 11 | 0.524 |
| 0.50 | 21/24 | 0.875 | 11 | 0.524 |
| 0.65 | 20/24 | 0.833 | 11 | 0.550 |
| 0.80 | 15/24 | 0.625 | 8 | 0.533 |

## Differential Review Candidates
| Case | Verifier | NLI label | Meaning |
| --- | --- | --- | --- |
| preserve_budget_digit_words | reject | entailment | possible_over_refusal |
| preserve_access_policy | ambiguous | entailment | review |
| preserve_direction_relationship | reject | entailment | possible_over_refusal |
| preserve_quantifier | ambiguous | entailment | review |
| flip_quantifier | ambiguous | contradiction | review |
| preserve_unit | ambiguous | entailment | review |
| flip_unit | ambiguous | contradiction | review |
| preserve_multiclause_scope | ambiguous | entailment | review |
| near_miss_same_topic | ambiguous | contradiction | review |
| preserve_currency | ambiguous | entailment | review |
| flip_currency | ambiguous | contradiction | review |
| neutral_related | reject | neutral | review |

## Detector Ablations
| Disabled detector | Baseline false accepts | Ablated false accepts | Newly accepted rejects |
| --- | ---: | ---: | --- |
| detect_entity_mismatch | 0 | 1 | flip_direction_relationship |
| detect_number_mismatch | 0 | 0 |  |
| detect_date_or_weekday_mismatch | 0 | 0 |  |
| detect_negation_flip | 0 | 1 | flip_rotation_negation |
| detect_antonym_flip_basic | 0 | 0 |  |
| detect_modal_flip | 0 | 0 |  |
