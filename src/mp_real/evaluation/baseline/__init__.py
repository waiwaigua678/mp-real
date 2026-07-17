"""Durable, reproducible real-robot evaluation baselines."""

from mp_real.evaluation.baseline.comparison import compare_baselines
from mp_real.evaluation.baseline.diff import BaselineDiff, BaselineDiffItem, diff_baselines
from mp_real.evaluation.baseline.models import Baseline, BaselineOpenLoopReference, BaselineRunReference
from mp_real.evaluation.baseline.service import BaselineConfigurationConflict, BaselineService
from mp_real.evaluation.baseline.store import BaselineNotFoundError, BaselineStore
from mp_real.evaluation.baseline.writer import BaselineReferenceWriter

__all__ = [
    "Baseline",
    "BaselineConfigurationConflict",
    "BaselineDiff",
    "BaselineDiffItem",
    "BaselineNotFoundError",
    "BaselineOpenLoopReference",
    "BaselineRunReference",
    "BaselineService",
    "BaselineStore",
    "BaselineReferenceWriter",
    "compare_baselines",
    "diff_baselines",
]
