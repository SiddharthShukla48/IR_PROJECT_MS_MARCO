"""Small cross-cutting helpers: logging, timing, memory hygiene, atomic IO, bounded parallel map."""
from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

log = logging.getLogger("msmarco_ir")
T = TypeVar("T")
R = TypeVar("R")


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, force=True,
        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    )


def configure_jvm(heap_gb: int | None) -> None:
    """Must run before pyserini is imported (the JVM reads this variable once at start-up)."""
    if heap_gb:
        os.environ["JAVA_TOOL_OPTIONS"] = f"-Xmx{heap_gb}g " + os.environ.get("JAVA_TOOL_OPTIONS", "")


@contextmanager
def timed(label: str, sink: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log.info("%s: %.1fs", label, dt)
    if sink is not None:
        sink[label] = dt


def free_memory() -> None:
    """Run at stage boundaries only (never inside loops): GC + hand freed pages back to the OS."""
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:  # pragma: no cover
        pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:  # glibc only; returns freed heap arenas to the OS after dropping huge Python objects
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj) -> None:
    """Atomic write: readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mark_complete(directory: Path) -> None:
    (directory / ".complete").touch()


def is_complete(directory: Path) -> bool:
    return (directory / ".complete").exists()


def ordered_parallel_map(executor, fn: Callable[[T], R], items: Iterable[T], max_inflight: int) -> Iterator[R]:
    """executor.map that (a) keeps result order and (b) never queues more than `max_inflight` items,
    so a generator over 8.8M passages is consumed lazily instead of being materialised in RAM."""
    pending: deque = deque()
    for item in items:
        pending.append(executor.submit(fn, item))
        if len(pending) >= max_inflight:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()
