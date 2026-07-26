"""Strict experiment suite for the ACM Transactions revision.

The package implements the revised experimental protocol requested in the
review comments: disjoint shadow/target pools, scalable client populations,
strict attack splits, identical-checkpoint contrasts, layer/full-update
attacks, adaptive defenses, and run-level statistics.
"""

from .data_protocol import ClientSpec, StrictPools

__all__ = ["ClientSpec", "StrictPools"]
