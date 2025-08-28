# FILE: scripts/profiling_tools.py
# =============================================================================
# 🌍 World Discovery Engine (WDE) — Profiling & Performance Tools
#
# Goals
# -----
# - Give developers a one-stop toolbox to measure runtime, CPU, memory, and (optionally)
#   GPU/torch metrics for any WDE stage or command.
# - Run standalone (CLI) or be imported by other modules (decorators / context managers).
# - Stdlib-first; gracefully enhance if optional deps (psutil, line_profiler, torch) exist.
#
# Features
# --------
# 1) High-resolution timers: Timer(), @timeit, time_block()
# 2) CPU profiling: cProfile runner (writes .prof + human summary + JSON top-N)
# 3) Memory profiling: tracemalloc snapshot + (optional) psutil RSS/USS reporting
# 4) Torch/CUDA snapshot (optional) if torch is installed
# 5) Subprocess wrapper: profile arbitrary shell command
# 6) Simple flamegraph guidance (py-spy) if installed; no hard dependency
#
# CLI
# ---
#   python scripts/profiling_tools.py cprofile \
#       --cmd "python -m world_engine.cli full-run --config configs/default.yaml" \
#       --out artifacts/profiles \
#       --topn 40
#
#   python scripts/profiling_tools.py memory \
#       --cmd "python -m world_engine.cli scan --config configs/default.yaml" \
#       --out artifacts/profiles
#
#   python scripts/profiling_tools.py timers \
#       --repeat 3 \
#       --cmd "python -m world_engine.cli verify --config configs/default.yaml"
#
# Import usage
# ------------
#   from scripts.profiling_tools import Timer, timeit, time_block, profile_block
#
#   @timeit(label="detect-stage")
#   def my_fn(...): ...
#
#   with time_block("calibration"):
#       ...
#
#   # CPU+memory profiling in a block (stdout + files):
#   from scripts.profiling_tools import Profiler
#   prof = Profiler(out_dir="artifacts/profiles")
#   with prof.profile_block("verify"):
#       run_verify(cfg)
#
# Outputs
# -------
# - {out}/cpu_{label}.prof        # raw cProfile
# - {out}/cpu_{label}.txt         # human-readable sorted stats
# - {out}/cpu_{label}.json        # machine-readable top-N
# - {out}/mem_{label}.json        # memory snapshot (tracemalloc + psutil if available)
# - {out}/timers.json             # structured wall-time records (when using timers CLI)
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import os
import pstats
import shlex
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# Optional deps
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # graceful degrade

try:
    import torch  # type: ignore
except Exception:
    torch = None  # graceful degrade

try:
    from line_profiler import LineProfiler  # type: ignore
except Exception:
    LineProfiler = None  # graceful degrade


# =============================================================================
# Utility: ensure output directory
# =============================================================================

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Timers
# =============================================================================

@dataclass
class TimeRecord:
    label: str
    wall_seconds: float
    started_at: float
    finished_at: float


class Timer:
    """High-resolution wall-time timer with context-manager convenience."""

    def __init__(self, label: str = "") -> None:
        self.label = label or "timer"
        self._t0 = 0.0
        self._t1 = 0.0
        self.elapsed = 0.0

    def start(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        self._t1 = time.perf_counter()
        self.elapsed = self._t1 - self._t0
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def time_block(label: str) -> Generator[Timer, None, None]:
    """Context manager for timing a code block."""
    t = Timer(label)
    with t:
        yield t


def timeit(label: Optional[str] = None) -> Callable[..., Any]:
    """
    Decorator to time a function; prints elapsed seconds to stdout.
    Example:
        @timeit("detect-stage")
        def run_detect(...):
            ...
    """
    def dec(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs):
            name = label or fn.__name__
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                print(f"[TIMER] {name}: {dt:.6f}s", flush=True)
        return wrapper
    return dec


# =============================================================================
# Memory Snapshot
# =============================================================================

def _psutil_mem_snapshot() -> Dict[str, Any]:
    if psutil is None:
        return {"psutil": False}
    proc = psutil.Process(os.getpid())
    try:
        with proc.oneshot():
            mem = proc.memory_full_info() if hasattr(proc, "memory_full_info") else proc.memory_info()
            rss = int(mem.rss)
            try:
                uss = int(mem.uss)  # some platforms
            except Exception:
                uss = None
            cpu = proc.cpu_percent(interval=0.0)
            return {
                "psutil": True,
                "rss_bytes": rss,
                "uss_bytes": uss,
                "cpu_percent": cpu,
                "num_threads": proc.num_threads(),
            }
    except Exception as e:
        return {"psutil": False, "error": str(e)}


def _torch_snapshot() -> Dict[str, Any]:
    if torch is None:
        return {"torch": False}
    snap: Dict[str, Any] = {"torch": True, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            snap["device_index"] = int(dev)
            snap["device_name"] = torch.cuda.get_device_name(dev)
            snap["mem_allocated_bytes"] = int(torch.cuda.memory_allocated(dev))
            snap["mem_reserved_bytes"] = int(torch.cuda.memory_reserved(dev))
            snap["max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(dev))
            snap["max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(dev))
        except Exception as e:
            snap["cuda_error"] = str(e)
    return snap


def memory_snapshot(label: str = "", top: int = 15) -> Dict[str, Any]:
    """Take a tracemalloc snapshot and return top allocators + psutil/torch info."""
    info: Dict[str, Any] = {"label": label, "timestamp": time.time()}
    # tracemalloc
    try:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            time.sleep(0)  # allow to start
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("lineno")
        top_stats = []
        for s in stats[:top]:
            top_stats.append(
                {
                    "trace": str(s.traceback.format()[-1]).strip(),
                    "size_bytes": int(s.size),
                    "count": int(s.count),
                }
            )
        info["tracemalloc"] = {
            "n_traces": int(len(stats)),
            "top": top_stats,
        }
    except Exception as e:
        info["tracemalloc_error"] = str(e)
    # psutil
    info["process"] = _psutil_mem_snapshot()
    # torch
    info["torch"] = _torch_snapshot()
    return info


# =============================================================================
# CPU Profiling (cProfile) + summaries
# =============================================================================

def _pstats_summary_str(prof_path: Path, sort: str = "cumulative", topn: int = 40) -> str:
    s = io.StringIO()
    ps = pstats.Stats(str(prof_path), stream=s)
    ps.strip_dirs().sort_stats(sort).print_stats(topn)
    return s.getvalue()


def _pstats_json_top(prof_path: Path, sort: str = "cumulative", topn: int = 40) -> Dict[str, Any]:
    ps = pstats.Stats(str(prof_path))
    ps.strip_dirs().sort_stats(sort)
    # Extract function stats (filename:lineno(function)) for top entries
    entries = []
    for fn_tuple, func_stats in list(ps.stats.items())[:]:  # type: ignore
        # func_stats: (cc, nc, tt, ct, callers)
        cc, nc, tt, ct, callers = func_stats
        entries.append(
            {
                "file": fn_tuple[0],
                "line": fn_tuple[1],
                "func": fn_tuple[2],
                "calls": int(cc),
                "primitive_calls": int(nc),
                "time_total": float(tt),
                "time_cumulative": float(ct),
            }
        )
    # Naive top-N by chosen key
    key = "time_cumulative" if sort == "cumulative" else "time_total"
    entries.sort(key=lambda x: x.get(key, 0.0), reverse=True)
    return {"sort": sort, "top": entries[:topn], "count": len(entries)}


def run_cprofile(label: str, cmd: Optional[str], out_dir: Union[str, Path], topn: int = 40) -> Dict[str, Any]:
    """
    Run cProfile over a shell command (subprocess) OR profile the current process block.
    If cmd is provided, we exec it under `python -m cProfile -o ...`.
    """
    out_dir = _ensure_dir(out_dir)
    prof_path = out_dir / f"cpu_{label}.prof"
    txt_path = out_dir / f"cpu_{label}.txt"
    json_path = out_dir / f"cpu_{label}.json"

    result: Dict[str, Any] = {"label": label, "cmd": cmd, "prof": str(prof_path)}

    if cmd:
        # Profile an external command
        # Use: python -m cProfile -o profile.prof <command args...>
        command = f"{shlex.quote(sys.executable)} -m cProfile -o {shlex.quote(str(prof_path))} {cmd}"
        t0 = time.perf_counter()
        rc = subprocess.call(command, shell=True)
        dt = time.perf_counter() - t0
        result.update({"returncode": rc, "elapsed": dt})
    else:
        # Profile a no-op block for demonstration (not commonly used in CLI path)
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        # ---- no-op / placeholder
        time.sleep(0.001)
        # ----
        pr.disable()
        dt = time.perf_counter() - t0
        pr.dump_stats(str(prof_path))
        result.update({"returncode": 0, "elapsed": dt})

    # Summaries
    try:
        txt = _pstats_summary_str(prof_path, "cumulative", topn=topn)
        (out_dir / f"cpu_{label}.txt").write_text(txt, encoding="utf-8")
        result["summary_path"] = str(txt_path)
    except Exception as e:
        result["summary_error"] = str(e)

    try:
        j = _pstats_json_top(prof_path, "cumulative", topn=topn)
        (out_dir / f"cpu_{label}.json").write_text(json.dumps(j, indent=2), encoding="utf-8")
        result["top_json_path"] = str(json_path)
    except Exception as e:
        result["json_error"] = str(e)

    return result


# =============================================================================
# Line Profiler (optional)
# =============================================================================

def run_line_prof(funcs: List[Callable], label: str, out_dir: Union[str, Path], *call_args, **call_kwargs) -> Dict[str, Any]:
    """
    Run line_profiler on a list of functions (if available).
    This is import-time only usage; typical developer flow: add your target functions and call here.
    """
    out_dir = _ensure_dir(out_dir)
    lp_txt = out_dir / f"line_{label}.txt"
    if LineProfiler is None:
        lp_txt.write_text("line_profiler not available. pip install line_profiler\n", encoding="utf-8")
        return {"available": False, "txt": str(lp_txt)}
    lp = LineProfiler()
    for f in funcs:
        lp.add_function(f)
    lp.enable_by_count()
    t0 = time.perf_counter()
    try:
        # Assumes first function is the entrypoint in funcs[0]
        ret = funcs[0](*call_args, **call_kwargs)
    finally:
        dt = time.perf_counter() - t0
        lp.disable()
    s = io.StringIO()
    lp.print_stats(stream=s)
    lp_txt.write_text(s.getvalue(), encoding="utf-8")
    return {"available": True, "elapsed": dt, "txt": str(lp_txt)}


# =============================================================================
# Profiler class for with-block CPU+Memory capture
# =============================================================================

class Profiler:
    """
    Combined profiler for a code block: cProfile (in-process) + memory snapshot before/after.

    Usage:
        prof = Profiler(out_dir="artifacts/profiles")
        with prof.profile_block("verify"):
            run_verify(cfg)

    Outputs:
        cpu_{label}.prof / cpu_{label}.txt / cpu_{label}.json
        mem_{label}.json
    """

    def __init__(self, out_dir: Union[str, Path]) -> None:
        self.out_dir = _ensure_dir(out_dir)
        self._pr: Optional[cProfile.Profile] = None
        self._label: str = ""
        self._t0: float = 0.0

    @contextlib.contextmanager
    def profile_block(self, label: str) -> Generator[None, None, None]:
        self._label = label
        prof_path = self.out_dir / f"cpu_{label}.prof"
        txt_path = self.out_dir / f"cpu_{label}.txt"
        json_path = self.out_dir / f"cpu_{label}.json"
        mem_path = self.out_dir / f"mem_{label}.json"

        # Memory snapshot BEFORE
        mem_before = memory_snapshot(label=f"{label}:before")

        # cProfile start
        self._pr = cProfile.Profile()
        self._t0 = time.perf_counter()
        self._pr.enable()
        try:
            yield
        finally:
            # stop cProfile
            self._pr.disable()
            elapsed = time.perf_counter() - self._t0
            self._pr.dump_stats(str(prof_path))

            # Memory snapshot AFTER
            mem_after = memory_snapshot(label=f"{label}:after")

            # Summaries
            try:
                txt = _pstats_summary_str(prof_path, "cumulative", topn=40)
                txt_path.write_text(txt, encoding="utf-8")
            except Exception as e:
                txt_path.write_text(f"Summary error: {e}\n", encoding="utf-8")

            try:
                j = _pstats_json_top(prof_path, "cumulative", topn=40)
                json_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
            except Exception as e:
                json_path.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")

            # Combine memory results
            mem_report = {
                "label": label,
                "elapsed": elapsed,
                "before": mem_before,
                "after": mem_after,
            }
            mem_path.write_text(json.dumps(mem_report, indent=2), encoding="utf-8")


# =============================================================================
# Subprocess helpers and CLI commands
# =============================================================================

def run_shell(cmd: str) -> int:
    """Run a shell command with inherited stdout/stderr; return exit code."""
    print(f"[RUN] {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def cli_cprofile(args: argparse.Namespace) -> int:
    label = args.label or "command"
    res = run_cprofile(label=label, cmd=args.cmd, out_dir=args.out, topn=args.topn)
    print(json.dumps(res, indent=2))
    return int(res.get("returncode", 0) != 0)


def cli_memory(args: argparse.Namespace) -> int:
    # memory snapshot around a command (coarse)
    out_dir = _ensure_dir(args.out)
    label = args.label or "command"
    before = memory_snapshot(label=f"{label}:before")
    rc = run_shell(args.cmd) if args.cmd else 0
    after = memory_snapshot(label=f"{label}:after")
    report = {"label": label, "returncode": rc, "before": before, "after": after}
    (out_dir / f"mem_{label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(rc != 0)


def cli_timers(args: argparse.Namespace) -> int:
    # Repeat command N times; collect wall-times
    out_dir = _ensure_dir(args.out)
    label = args.label or "command"
    records: List[Dict[str, Any]] = []
    rcs = []
    for i in range(max(1, int(args.repeat))):
        t0 = time.perf_counter()
        rc = run_shell(args.cmd) if args.cmd else 0
        dt = time.perf_counter() - t0
        records.append(asdict(TimeRecord(label=f"{label}#{i+1}", wall_seconds=dt, started_at=t0, finished_at=t0+dt)))
        rcs.append(rc)
    (out_dir / "timers.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"records": records}, indent=2))
    return int(any(rc != 0 for rc in rcs))


def cli_pyspy_hint(args: argparse.Namespace) -> int:
    """
    Print guidance for using py-spy (if installed) to generate a flamegraph without code changes.
    """
    print("# py-spy quickstart (requires: pip install py-spy)\n")
    print("py-spy top -- python -m world_engine.cli full-run --config configs/default.yaml")
    print("\n# Flamegraph SVG (sampling profiler):")
    print("py-spy record -o artifacts/profiles/flame.svg --rate 250 -- ")
    print("    python -m world_engine.cli verify --config configs/default.yaml")
    print("\n# While running another process by PID:")
    print("py-spy dump --pid <PID> --native -o artifacts/profiles/stack.txt")
    return 0


# =============================================================================
# Argparse top-level
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="WDE Profiling Tools • cProfile / memory / timers / py-spy hints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    # cprofile
    p_cprof = sub.add_parser("cprofile", help="Run cProfile over a shell command.")
    p_cprof.add_argument("--cmd", type=str, required=True, help="Shell command to profile.")
    p_cprof.add_argument("--label", type=str, default="command", help="Label for outputs.")
    p_cprof.add_argument("--out", type=str, default="artifacts/profiles", help="Output directory.")
    p_cprof.add_argument("--topn", type=int, default=40, help="Top-N functions in summary.")
    p_cprof.set_defaults(func=cli_cprofile)

    # memory
    p_mem = sub.add_parser("memory", help="Take tracemalloc+psutil snapshots around a command.")
    p_mem.add_argument("--cmd", type=str, required=True, help="Shell command to run.")
    p_mem.add_argument("--label", type=str, default="command", help="Label for outputs.")
    p_mem.add_argument("--out", type=str, default="artifacts/profiles", help="Output directory.")
    p_mem.set_defaults(func=cli_memory)

    # timers
    p_t = sub.add_parser("timers", help="Repeat & measure wall-clock for a command.")
    p_t.add_argument("--cmd", type=str, required=True, help="Shell command to run.")
    p_t.add_argument("--label", type=str, default="command", help="Label for this measurement set.")
    p_t.add_argument("--repeat", type=int, default=3, help="Number of repetitions.")
    p_t.add_argument("--out", type=str, default="artifacts/profiles", help="Output directory.")
    p_t.set_defaults(func=cli_timers)

    # py-spy hint
    p_h = sub.add_parser("pyspy", help="Print py-spy usage hints (no dependencies).")
    p_h.set_defaults(func=cli_pyspy_hint)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
