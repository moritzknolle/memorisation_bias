"""The audit itself: from a set of leave-many-out runs to a per-record verdict.

`core` holds the whole pipeline -- loading each run's logits and training-subset mask,
partitioning a record's predictions into IN (models that trained on this patient) and OUT
(models that did not), and testing the two groups for a difference with FDR correction
across records.
"""
