from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mp_real.pose.models import (
    MappingEntry,
    PoseMappingConfig,
    PoseSafetyLimits,
    PoseValidationIssue,
    PoseValidationReport,
    RecordedPoseTarget,
    ValidatedPoseTarget,
)
from mp_real.runtime.models import ActionSpec


class MoveToStateValidator:
    """Strictly map a recorded state into one concrete robot state schema.

    There is deliberately no positional fallback.  A different robot name,
    field order, unit, or semantic annotation needs an explicit mapping config.
    """

    def __init__(
        self,
        robot_name: str,
        action_spec: ActionSpec,
        *,
        mapping_config: PoseMappingConfig | None = None,
        safety_limits: PoseSafetyLimits | None = None,
    ) -> None:
        self.robot_name = robot_name
        self.action_spec = action_spec
        self.mapping_config = mapping_config
        self.safety_limits = safety_limits

    def validate(self, target: RecordedPoseTarget) -> ValidatedPoseTarget:
        issues: list[PoseValidationIssue] = []
        warnings: list[PoseValidationIssue] = []
        metadata_status = str(target.source_metadata.get("dataset_status", "complete")).lower()
        if metadata_status != "complete":
            issues.append(
                PoseValidationIssue("dataset_not_complete", f"dataset status is {metadata_status!r}, not complete")
            )
        if target.state_values.shape != (target.action_spec.state_dim,):
            issues.append(
                PoseValidationIssue("state_dimension_invalid", "recorded state does not match its ActionSpec")
            )
        if not np.isfinite(target.state_values).all():
            issues.append(PoseValidationIssue("state_not_finite", "recorded state contains NaN or Inf"))

        if self.mapping_config is None:
            values, mappings = self._validate_identity(target, issues)
            mapping_fingerprint = None
        else:
            values, mappings = self._validate_explicit_mapping(target, issues)
            mapping_fingerprint = self.mapping_config.fingerprint

        if self.safety_limits is None:
            warnings.append(
                PoseValidationIssue(
                    "joint_limits_not_provided", "generic joint limits require vendor validation", severity="warning"
                )
            )
        elif self.safety_limits.lower.shape != values.shape:
            issues.append(
                PoseValidationIssue("joint_limit_dimension_mismatch", "joint limits do not match robot state dimension")
            )
        else:
            below = np.flatnonzero(values < self.safety_limits.lower)
            above = np.flatnonzero(values > self.safety_limits.upper)
            for index in (*below, *above):
                issues.append(
                    PoseValidationIssue(
                        "joint_limit_exceeded",
                        f"target {self.action_spec.state_field_names[index]!r} is outside configured joint limits",
                        int(index),
                    )
                )

        report = PoseValidationReport(tuple(issues), tuple(warnings), mapping_fingerprint)
        return ValidatedPoseTarget(
            values=values,
            field_names=self.action_spec.state_field_names,
            gripper_indices=tuple(
                index
                for index, field in enumerate(self.action_spec.state_fields)
                if field.semantics == "gripper_open_fraction"
            ),
            mappings=tuple(mappings),
            report=report,
        )

    def _validate_identity(
        self, target: RecordedPoseTarget, issues: list[PoseValidationIssue]
    ) -> tuple[np.ndarray, Sequence[MappingEntry]]:
        if target.robot_name != self.robot_name:
            issues.append(
                PoseValidationIssue(
                    "robot_name_mismatch",
                    f"recorded robot {target.robot_name!r} does not match connected robot {self.robot_name!r}",
                )
            )
        if target.action_spec.state_dim != self.action_spec.state_dim:
            issues.append(PoseValidationIssue("state_dimension_mismatch", "recorded and robot state dimensions differ"))
            return np.zeros(self.action_spec.state_dim, dtype=np.float32), ()
        if target.state_schema != self.action_spec.state_field_names:
            issues.append(
                PoseValidationIssue("joint_name_or_order_mismatch", "state field names/order require explicit mapping")
            )
        mappings: list[MappingEntry] = []
        for index, (source, destination) in enumerate(zip(target.state_fields, self.action_spec.state_fields)):
            if source.unit != destination.unit:
                issues.append(PoseValidationIssue("joint_unit_mismatch", f"field {source.name!r} unit differs", index))
            if source.semantics != destination.semantics:
                issues.append(
                    PoseValidationIssue("field_semantics_mismatch", f"field {source.name!r} semantics differ", index)
                )
            mappings.append(
                MappingEntry(
                    source.name,
                    destination.name,
                    source_unit=source.unit,
                    target_unit=destination.unit,
                    semantics=source.semantics,
                )
            )
        return target.state_values.copy(), mappings

    def _validate_explicit_mapping(
        self, target: RecordedPoseTarget, issues: list[PoseValidationIssue]
    ) -> tuple[np.ndarray, Sequence[MappingEntry]]:
        assert self.mapping_config is not None
        config = self.mapping_config
        if config.source_robot_name != target.robot_name or config.target_robot_name != self.robot_name:
            issues.append(
                PoseValidationIssue(
                    "mapping_robot_name_mismatch",
                    "mapping config source/target robot names must exactly match this transfer",
                )
            )
        source_by_name = {field.name: (index, field) for index, field in enumerate(target.state_fields)}
        target_by_name = {field.name: (index, field) for index, field in enumerate(self.action_spec.state_fields)}
        mapped_source = {entry.source_name for entry in config.entries}
        mapped_target = {entry.target_name for entry in config.entries}
        if mapped_source != set(source_by_name) or mapped_target != set(target_by_name):
            issues.append(
                PoseValidationIssue(
                    "mapping_not_total", "mapping config must map every source and robot state field exactly once"
                )
            )
        values = np.zeros(self.action_spec.state_dim, dtype=np.float32)
        for entry in config.entries:
            source_item = source_by_name.get(entry.source_name)
            destination_item = target_by_name.get(entry.target_name)
            if source_item is None or destination_item is None:
                issues.append(
                    PoseValidationIssue(
                        "mapping_field_missing",
                        f"mapping field {entry.source_name!r}->{entry.target_name!r} does not exist",
                    )
                )
                continue
            source_index, source_field = source_item
            destination_index, destination_field = destination_item
            expected_source_unit = entry.source_unit or source_field.unit
            expected_target_unit = entry.target_unit or destination_field.unit
            expected_semantics = entry.semantics or source_field.semantics
            if source_field.unit not in {expected_source_unit, "unknown"}:
                issues.append(
                    PoseValidationIssue(
                        "mapping_source_unit_mismatch", f"source unit mismatch for {entry.source_name!r}", source_index
                    )
                )
            if destination_field.unit != expected_target_unit:
                issues.append(
                    PoseValidationIssue(
                        "mapping_target_unit_mismatch",
                        f"target unit mismatch for {entry.target_name!r}",
                        destination_index,
                    )
                )
            if source_field.semantics not in {expected_semantics, "unknown"}:
                issues.append(
                    PoseValidationIssue(
                        "mapping_source_semantics_mismatch",
                        f"source semantics mismatch for {entry.source_name!r}",
                        source_index,
                    )
                )
            if destination_field.semantics != expected_semantics:
                issues.append(
                    PoseValidationIssue(
                        "mapping_semantics_mismatch", f"semantics mismatch for {entry.target_name!r}", destination_index
                    )
                )
            if expected_source_unit != expected_target_unit and entry.scale == 1.0 and entry.offset == 0.0:
                issues.append(
                    PoseValidationIssue(
                        "mapping_unit_conversion_missing",
                        f"unit conversion missing for {entry.source_name!r}",
                        source_index,
                    )
                )
            values[destination_index] = target.state_values[source_index] * entry.scale + entry.offset
        return values, tuple(config.entries)
