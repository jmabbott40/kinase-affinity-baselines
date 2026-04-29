"""kinase-affinity-baselines: kinase-specific application using target-affinity-ml.

This package is FROZEN at v1.0 — the version accompanying the published
preprint. Ongoing work has moved to:
- Library: target-affinity-ml (https://github.com/jmabbott40/target-affinity-ml)
- Phase 1 application: gpcr-aminergic-benchmarks (https://github.com/jmabbott40/gpcr-aminergic-benchmarks)

For backward compatibility, all submodules re-export from the library:
    from kinase_affinity.models import RandomForestModel  # → target_affinity_ml.models.RandomForestModel
"""
import sys as _sys

__version__ = "1.0.0"

# Backward-compatibility re-exports from the library.
# We register the library's subpackages under kinase_affinity.* in sys.modules
# so that `from kinase_affinity.X import Y` and `from kinase_affinity.X.Z import W`
# continue to work for downstream code (notebooks, scripts, tests) that was
# written against the old kinase_affinity layout.
from target_affinity_ml import (
    data,
    features,
    models,
    training,
    evaluation,
    visualization,
)

for _name, _module in [
    ("data", data),
    ("features", features),
    ("models", models),
    ("training", training),
    ("evaluation", evaluation),
    ("visualization", visualization),
]:
    _sys.modules[f"kinase_affinity.{_name}"] = _module
    # Also register any already-imported child modules (e.g. data.splits) so
    # `from kinase_affinity.data.splits import random_split` resolves.
    _prefix = f"target_affinity_ml.{_name}."
    for _full_name, _child in list(_sys.modules.items()):
        if _full_name.startswith(_prefix):
            _suffix = _full_name[len(_prefix):]
            _sys.modules[f"kinase_affinity.{_name}.{_suffix}"] = _child

del _sys, _name, _module, _prefix, _full_name, _child, _suffix

__all__ = [
    "data", "features", "models", "training", "evaluation", "visualization",
    "__version__",
]
