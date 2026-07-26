"""Strict experiment suite for the ACM Transactions revision.

The package implements the revised experimental protocol requested in the
review comments: disjoint shadow/target pools, scalable client populations,
strict attack splits, identical-checkpoint contrasts, layer/full-update
attacks, adaptive defenses, and run-level statistics.
"""

from . import data_protocol as _data_protocol
from .data_protocol import ClientSpec, StrictPools
from .strict_split import build_strict_pools as _strict_build_strict_pools


def _compat_build_strict_pools(*args, **kwargs):
    """Route legacy prepare-data calls to the final strict split implementation.

    Older CLI code used ``deduplicate`` and ``enable_near_duplicate_check``.
    In the revision protocol these mean exact-pair deduplication and grouping of
    near-duplicate families, respectively.  We also infer joint image/text label
    stratification when both modality labels are available.
    """
    if "deduplicate" in kwargs:
        kwargs.setdefault("exact_deduplicate", bool(kwargs.pop("deduplicate")))
    if "enable_near_duplicate_check" in kwargs:
        kwargs.setdefault("group_near_duplicates", bool(kwargs.pop("enable_near_duplicate_check")))

    if "stratify_keys" not in kwargs and args:
        train_data = args[0]
        if train_data and all("image_label" in x and "text_label" in x for x in train_data):
            kwargs["stratify_keys"] = ["image_label", "text_label"]
        elif train_data and all("label" in x for x in train_data):
            kwargs["stratify_keys"] = ["label"]

    return _strict_build_strict_pools(*args, **kwargs)


# Keep ``python -m src.acm_revision.cli prepare-data`` backward compatible while
# ensuring it cannot accidentally use the obsolete near-duplicate deletion path.
_data_protocol.build_strict_pools = _compat_build_strict_pools

__all__ = ["ClientSpec", "StrictPools"]
