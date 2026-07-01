"""Defense utilities for counterfactual association-sensitive subspace filtering."""

from .subspace import (
    CounterfactualSubspace,
    fit_counterfactual_subspace,
    project_onto_subspace,
    remove_subspace_component,
    subspace_similarity,
    principal_angles_degrees,
)

__all__ = [
    "CounterfactualSubspace",
    "fit_counterfactual_subspace",
    "project_onto_subspace",
    "remove_subspace_component",
    "subspace_similarity",
    "principal_angles_degrees",
]
