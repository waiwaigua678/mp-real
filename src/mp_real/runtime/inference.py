from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from mp_real.common.runtime import RealTimeChunkingBuffer, raise_rtc_producer_error, select_rtc_cursor
from mp_real.common.runtime import sleep_remaining
from mp_real.runtime.config import InferenceLoopConfig


class PolicyClient(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


class InferenceAdapter(Protocol):
    name: str

    def observe(self) -> dict[str, Any]: ...

    def decode_action_chunk(self, response: dict[str, Any], replan_steps: int) -> np.ndarray: ...

    def initial_action(self) -> np.ndarray: ...

    def stabilize_action(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray: ...

    def execute_transition(self, previous: np.ndarray | None, target: np.ndarray) -> np.ndarray: ...

    def infer_only_metadata(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def profile(self, stage: str, elapsed_s: float) -> None: ...

    def infer_only_interval_s(self) -> float: ...


def _fetch_chunk(client: PolicyClient, adapter: InferenceAdapter, config: InferenceLoopConfig) -> tuple[dict[str, Any], np.ndarray]:
    observation_started = time.monotonic()
    observation = adapter.observe()
    adapter.profile("observation", time.monotonic() - observation_started)
    infer_started = time.monotonic()
    response = client.infer(observation)
    adapter.profile("inference", time.monotonic() - infer_started)
    return observation, adapter.decode_action_chunk(response, config.replan_steps)


def run_infer_only(client: PolicyClient, adapter: InferenceAdapter, config: InferenceLoopConfig) -> None:
    chunks: list[np.ndarray] = []
    states: list[np.ndarray] = []
    extras: dict[str, list[Any]] = {}
    for index in range(config.infer_only_chunks):
        observation, chunk = _fetch_chunk(client, adapter, config)
        chunks.append(chunk)
        states.append(np.asarray(observation["state"], dtype=np.float32))
        for key, value in adapter.infer_only_metadata(observation).items():
            extras.setdefault(key, []).append(value)
        print(f"action_chunk[{index}] shape={chunk.shape}")
        print(np.array2string(chunk, precision=5, suppress_small=True))
        if index + 1 < config.infer_only_chunks:
            interval_s = adapter.infer_only_interval_s()
            if interval_s > 0:
                time.sleep(interval_s)

    if config.infer_only_output is None:
        return
    config.infer_only_output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "actions": np.stack(chunks, axis=0),
        "states": np.stack(states, axis=0),
        "prompt": np.asarray(config.prompt),
    }
    payload.update({key: np.asarray(values, dtype=object) for key, values in extras.items()})
    np.savez_compressed(config.infer_only_output, **payload)
    logging.info("Saved infer-only action chunks to %s", config.infer_only_output)


def run_sync_loop(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> None:
    plan: collections.deque[np.ndarray] = collections.deque()
    dt = 1.0 / config.fps
    step = 0
    previous: np.ndarray | None = adapter.initial_action()
    logging.info("Starting %s synchronous inference loop", adapter.name)
    while (stop_event is None or not stop_event.is_set()) and (config.max_steps is None or step < config.max_steps):
        loop_started = time.monotonic()
        if not plan:
            _, chunk = _fetch_chunk(client, adapter, config)
            plan.extend(chunk)
        action = adapter.stabilize_action(plan.popleft(), previous)
        execute_started = time.monotonic()
        previous = adapter.execute_transition(previous, action)
        adapter.profile("execution", time.monotonic() - execute_started)
        step += 1
        if on_step is not None:
            on_step(step, len(plan))
        if config.log_timing and step % 10 == 0:
            logging.info("sync step=%d loop=%.3fs queued=%d", step, time.monotonic() - loop_started, len(plan))
        sleep_remaining(loop_started, dt)


def _rtc_producer(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    buffer: RealTimeChunkingBuffer,
    stop_event: threading.Event,
    errors: queue.Queue[BaseException],
) -> None:
    while not stop_event.is_set():
        cursor = select_rtc_cursor(buffer, config)
        if cursor is None:
            stop_event.wait(min(0.005, 0.25 / config.fps))
            continue
        generation = buffer.get_generation()
        try:
            started = time.monotonic()
            _, chunk = _fetch_chunk(client, adapter, config)
            accepted = buffer.enqueue(chunk, cursor, generation)
            if config.log_timing:
                logging.info(
                    "RTC producer robot=%s cursor=%d accepted=%s chunk=%s elapsed=%.3fs",
                    adapter.name,
                    cursor,
                    accepted,
                    chunk.shape,
                    time.monotonic() - started,
                )
        except Exception as exc:
            errors.put(exc)
            stop_event.set()
            return


def run_rtc_loop(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> None:
    buffer = RealTimeChunkingBuffer(exp_weight=config.rtc_exp_weight)
    buffer.clear()
    stop_event = stop_event or threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()
    producer = threading.Thread(
        target=_rtc_producer,
        name=f"{adapter.name}-rtc-action-producer",
        args=(client, adapter, config, buffer, stop_event, errors),
        daemon=True,
    )
    producer.start()
    dt = 1.0 / config.fps
    step = 0
    previous: np.ndarray | None = adapter.initial_action()
    last_wait_log = 0.0
    logging.info("Starting %s RTC inference loop", adapter.name)
    try:
        while not stop_event.is_set() and (config.max_steps is None or step < config.max_steps):
            loop_started = time.monotonic()
            raise_rtc_producer_error(errors)
            buffer.set_control_time(step)
            action = buffer.get_action(step)
            if action is None:
                if previous is not None and config.hold_last_action:
                    previous = adapter.execute_transition(previous, previous)
                now = time.monotonic()
                if now - last_wait_log > 1.0:
                    logging.warning("Waiting for RTC action at control step %d", step)
                    last_wait_log = now
                sleep_remaining(loop_started, dt)
                continue
            action = adapter.stabilize_action(action, previous)
            execute_started = time.monotonic()
            previous = adapter.execute_transition(previous, action)
            adapter.profile("execution", time.monotonic() - execute_started)
            step += 1
            if on_step is not None:
                on_step(step, 0)
            if config.log_timing and step % 10 == 0:
                logging.info("rtc step=%d loop=%.3fs", step, time.monotonic() - loop_started)
            sleep_remaining(loop_started, dt)
    finally:
        stop_event.set()
        producer.join(timeout=2.0)
        raise_rtc_producer_error(errors)


def run_policy_loop(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> None:
    config.validate()
    if config.infer_only:
        run_infer_only(client, adapter, config)
    elif config.use_rtc:
        run_rtc_loop(client, adapter, config, stop_event=stop_event, on_step=on_step)
    else:
        run_sync_loop(client, adapter, config, stop_event=stop_event, on_step=on_step)
