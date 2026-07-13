from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from mp_real.common.runtime import (
    RealTimeChunkingBuffer,
    raise_rtc_producer_error,
    select_rtc_cursor,
    sleep_remaining,
)
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


class InferenceHooks:
    """Non-blocking extension points for observing a policy loop in memory."""

    def on_loop_started(self, mode: str, config: InferenceLoopConfig) -> None:
        del mode, config

    def on_observation(self, observation: Mapping[str, Any]) -> None:
        del observation

    def on_inference_started(self, observation: Mapping[str, Any]) -> None:
        del observation

    def on_inference_finished(self, response: Mapping[str, Any], elapsed_s: float) -> None:
        del response, elapsed_s

    def on_chunk_received(self, chunk: np.ndarray) -> None:
        del chunk

    def on_action_selected(self, step: int, action: np.ndarray) -> None:
        del step, action

    def on_action_stabilized(self, step: int, action: np.ndarray) -> None:
        del step, action

    def on_action_executed(self, step: int, action: np.ndarray) -> None:
        del step, action

    def on_loop_stopped(self, mode: str) -> None:
        del mode

    def on_error(self, error: BaseException) -> None:
        del error


def _active_hooks(hooks: InferenceHooks | None) -> InferenceHooks:
    return hooks if hooks is not None else InferenceHooks()


def _fetch_chunk(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    hooks: InferenceHooks,
) -> tuple[dict[str, Any], np.ndarray]:
    observation_started_ns = time.monotonic_ns()
    observation = adapter.observe()
    adapter.profile("observation", (time.monotonic_ns() - observation_started_ns) / 1e9)
    hooks.on_observation(observation)
    hooks.on_inference_started(observation)
    infer_started_ns = time.monotonic_ns()
    response = client.infer(observation)
    infer_elapsed_s = (time.monotonic_ns() - infer_started_ns) / 1e9
    adapter.profile("inference", infer_elapsed_s)
    hooks.on_inference_finished(response, infer_elapsed_s)
    chunk = adapter.decode_action_chunk(response, config.replan_steps)
    hooks.on_chunk_received(chunk.copy())
    return observation, chunk


def run_infer_only(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
    hooks: InferenceHooks | None = None,
    print_chunks: bool = True,
) -> None:
    hooks = _active_hooks(hooks)
    chunks: list[np.ndarray] = []
    states: list[np.ndarray] = []
    extras: dict[str, list[Any]] = {}
    mode = "infer_only"
    try:
        hooks.on_loop_started(mode, config)
        for index in range(config.infer_only_chunks):
            if stop_event is not None and stop_event.is_set():
                break
            observation, chunk = _fetch_chunk(client, adapter, config, hooks)
            chunks.append(chunk)
            states.append(np.asarray(observation["state"], dtype=np.float32))
            for key, value in adapter.infer_only_metadata(observation).items():
                extras.setdefault(key, []).append(value)
            if print_chunks:
                print(f"action_chunk[{index}] shape={chunk.shape}")
                print(np.array2string(chunk, precision=5, suppress_small=True))
            if on_step is not None:
                on_step(index + 1, len(chunk))
            if index + 1 < config.infer_only_chunks:
                interval_s = adapter.infer_only_interval_s()
                if interval_s > 0:
                    if stop_event is None:
                        time.sleep(interval_s)
                    elif stop_event.wait(interval_s):
                        break

        if config.infer_only_output is None or not chunks:
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
    except BaseException as exc:
        hooks.on_error(exc)
        raise
    finally:
        hooks.on_loop_stopped(mode)


def run_sync_loop(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
    hooks: InferenceHooks | None = None,
) -> None:
    hooks = _active_hooks(hooks)
    plan: collections.deque[np.ndarray] = collections.deque()
    dt = 1.0 / config.fps
    step = 0
    mode = "sync"
    try:
        hooks.on_loop_started(mode, config)
        previous: np.ndarray | None = adapter.initial_action()
        logging.info("Starting %s synchronous inference loop", adapter.name)
        while (stop_event is None or not stop_event.is_set()) and (
            config.max_steps is None or step < config.max_steps
        ):
            loop_started = time.monotonic()
            if not plan:
                _, chunk = _fetch_chunk(client, adapter, config, hooks)
                plan.extend(chunk)
            selected = plan.popleft()
            hooks.on_action_selected(step, selected.copy())
            action = adapter.stabilize_action(selected, previous)
            hooks.on_action_stabilized(step, action.copy())
            execute_started_ns = time.monotonic_ns()
            previous = adapter.execute_transition(previous, action)
            adapter.profile("execution", (time.monotonic_ns() - execute_started_ns) / 1e9)
            hooks.on_action_executed(step, previous.copy())
            step += 1
            if on_step is not None:
                on_step(step, len(plan))
            if config.log_timing and step % 10 == 0:
                logging.info("sync step=%d loop=%.3fs queued=%d", step, time.monotonic() - loop_started, len(plan))
            sleep_remaining(loop_started, dt)
    except BaseException as exc:
        hooks.on_error(exc)
        raise
    finally:
        hooks.on_loop_stopped(mode)


def _rtc_producer(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    buffer: RealTimeChunkingBuffer,
    stop_event: threading.Event,
    errors: queue.Queue[BaseException],
    hooks: InferenceHooks,
) -> None:
    while not stop_event.is_set():
        cursor = select_rtc_cursor(buffer, config)
        if cursor is None:
            stop_event.wait(min(0.005, 0.25 / config.fps))
            continue
        generation = buffer.get_generation()
        try:
            started = time.monotonic()
            _, chunk = _fetch_chunk(client, adapter, config, hooks)
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
    hooks: InferenceHooks | None = None,
    producer_daemon: bool = True,
) -> None:
    hooks = _active_hooks(hooks)
    buffer = RealTimeChunkingBuffer(exp_weight=config.rtc_exp_weight)
    buffer.clear()
    stop_event = stop_event or threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()
    producer = threading.Thread(
        target=_rtc_producer,
        name=f"{adapter.name}-rtc-action-producer",
        args=(client, adapter, config, buffer, stop_event, errors, hooks),
        daemon=producer_daemon,
    )
    dt = 1.0 / config.fps
    step = 0
    last_wait_log = 0.0
    mode = "rtc"
    producer_started = False
    try:
        hooks.on_loop_started(mode, config)
        producer.start()
        producer_started = True
        previous: np.ndarray | None = adapter.initial_action()
        logging.info("Starting %s RTC inference loop", adapter.name)
        while not stop_event.is_set() and (config.max_steps is None or step < config.max_steps):
            loop_started = time.monotonic()
            raise_rtc_producer_error(errors)
            buffer.set_control_time(step)
            action = buffer.get_action(step)
            if action is None:
                if previous is not None and config.hold_last_action:
                    previous = adapter.execute_transition(previous, previous)
                    hooks.on_action_executed(step, previous.copy())
                now = time.monotonic()
                if now - last_wait_log > 1.0:
                    logging.warning("Waiting for RTC action at control step %d", step)
                    last_wait_log = now
                sleep_remaining(loop_started, dt)
                continue
            hooks.on_action_selected(step, action.copy())
            action = adapter.stabilize_action(action, previous)
            hooks.on_action_stabilized(step, action.copy())
            execute_started_ns = time.monotonic_ns()
            previous = adapter.execute_transition(previous, action)
            adapter.profile("execution", (time.monotonic_ns() - execute_started_ns) / 1e9)
            hooks.on_action_executed(step, previous.copy())
            step += 1
            if on_step is not None:
                on_step(step, 0)
            if config.log_timing and step % 10 == 0:
                logging.info("rtc step=%d loop=%.3fs", step, time.monotonic() - loop_started)
            sleep_remaining(loop_started, dt)
    except BaseException as exc:
        hooks.on_error(exc)
        raise
    finally:
        stop_event.set()
        if producer_started:
            client_timeout = getattr(client, "timeout", None)
            join_timeout = 2.0 if client_timeout is None else max(2.0, float(client_timeout) + 1.0)
            producer.join(timeout=join_timeout)
            if producer.is_alive() and (client_timeout is not None or not producer_daemon):
                errors.put(RuntimeError(f"RTC action producer did not stop within {join_timeout:.1f}s"))
        try:
            raise_rtc_producer_error(errors)
        except BaseException as exc:
            hooks.on_error(exc)
            raise
        finally:
            hooks.on_loop_stopped(mode)


def run_policy_loop(
    client: PolicyClient,
    adapter: InferenceAdapter,
    config: InferenceLoopConfig,
    *,
    stop_event: threading.Event | None = None,
    on_step: Callable[[int, int], None] | None = None,
    hooks: InferenceHooks | None = None,
    print_infer_only_chunks: bool = True,
    rtc_producer_daemon: bool = True,
) -> None:
    config.validate()
    if config.infer_only:
        run_infer_only(
            client,
            adapter,
            config,
            stop_event=stop_event,
            on_step=on_step,
            hooks=hooks,
            print_chunks=print_infer_only_chunks,
        )
    elif config.use_rtc:
        run_rtc_loop(
            client,
            adapter,
            config,
            stop_event=stop_event,
            on_step=on_step,
            hooks=hooks,
            producer_daemon=rtc_producer_daemon,
        )
    else:
        run_sync_loop(client, adapter, config, stop_event=stop_event, on_step=on_step, hooks=hooks)
