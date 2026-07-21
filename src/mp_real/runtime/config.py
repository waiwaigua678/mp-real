from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class InferenceLoopConfig:
    """Robot-independent policy-loop settings."""

    fps: float
    replan_steps: int
    max_steps: int | None
    use_rtc: bool
    rtc_replan_stride: int
    rtc_prefetch_steps: int
    rtc_exp_weight: float
    hold_last_action: bool
    infer_only: bool
    infer_only_chunks: int
    infer_only_output: Any
    prompt: str
    log_timing: bool

    @classmethod
    def from_args(cls, args: Any) -> InferenceLoopConfig:
        return cls(
            fps=args.fps,
            replan_steps=args.replan_steps,
            max_steps=args.max_steps,
            use_rtc=args.use_rtc,
            rtc_replan_stride=args.rtc_replan_stride,
            rtc_prefetch_steps=args.rtc_prefetch_steps,
            rtc_exp_weight=args.rtc_exp_weight,
            hold_last_action=args.hold_last_action,
            infer_only=args.infer_only,
            infer_only_chunks=args.infer_only_chunks,
            infer_only_output=args.infer_only_output,
            prompt=args.prompt,
            log_timing=args.log_timing,
        )

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.infer_only_chunks <= 0:
            raise ValueError("infer_only_chunks must be positive")
        if self.rtc_exp_weight < 0:
            raise ValueError("rtc_exp_weight must be non-negative")
        if self.rtc_replan_stride < 0 or self.rtc_prefetch_steps < 0:
            raise ValueError("RTC stride and prefetch steps must be non-negative")
        stride = self.replan_steps if self.rtc_replan_stride <= 0 else self.rtc_replan_stride
        if stride > self.replan_steps:
            raise ValueError("rtc_replan_stride must be <= replan_steps")
