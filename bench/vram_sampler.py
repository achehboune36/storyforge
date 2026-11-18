"""E0 — resource sampler.

Samples GPU, RAM and swap at a fixed rate to CSV. Every other experiment runs
with one of these attached.

Two GPU sensors are recorded deliberately:
  * nvidia-smi          — the whole device, including the Windows desktop
  * ComfyUI /system_stats — torch's own accounting inside ComfyUI

The divergence between them is diagnostic: if nvidia-smi pins near the device
total while torch reports free memory, the driver is spilling to host RAM. On
WSL that is silent and catastrophic for throughput, so it must be detected
rather than assumed away.

RAM and swap are sampled because loading multi-GB GGUF files through a 23 GiB
cgroup is what exhausted swap in the first place.

Usage:
    python -m bench.vram_sampler --out bench/results/e1.csv --label klein9b
    (Ctrl-C, or SIGTERM from a harness, to stop cleanly.)
"""

from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFY = "http://127.0.0.1:8188"
FIELDS = [
    "t",              # seconds since sampler start, 3dp
    "wall",           # unix time
    "label",
    "smi_used_mib",   # whole-device, nvidia-smi
    "smi_free_mib",
    "torch_free_mib",  # ComfyUI /system_stats devices[0].vram_free
    "torch_total_mib",
    "ram_used_mib",
    "ram_avail_mib",
    "swap_used_mib",
    "swap_free_mib",
]


def _nvidia_smi() -> tuple[int | None, int | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        used, free = (int(x.strip()) for x in out.split(","))
        return used, free
    except Exception:
        return None, None


def _comfy_stats() -> tuple[int | None, int | None]:
    """torch-scoped VRAM. None when ComfyUI is down — which is itself a datum."""
    try:
        with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=3) as r:
            import json
            dev = json.load(r)["devices"][0]
            return dev["vram_free"] // 1048576, dev["vram_total"] // 1048576
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
        return None, None


def _meminfo() -> tuple[int, int, int, int]:
    """(ram_used, ram_avail, swap_used, swap_free) in MiB, from /proc/meminfo."""
    vals: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                vals[k] = int(rest.split()[0]) // 1024
    ram_used = vals["MemTotal"] - vals["MemAvailable"]
    swap_used = vals["SwapTotal"] - vals["SwapFree"]
    return ram_used, vals["MemAvailable"], swap_used, vals["SwapFree"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--hz", type=float, default=2.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / args.hz
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    t0 = time.time()
    peak_smi = 0
    n = 0

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        while not stop:
            tick = time.time()
            smi_used, smi_free = _nvidia_smi()
            torch_free, torch_total = _comfy_stats()
            ram_u, ram_a, swap_u, swap_f = _meminfo()

            if smi_used is not None:
                peak_smi = max(peak_smi, smi_used)

            w.writerow({
                "t": f"{tick - t0:.3f}", "wall": f"{tick:.3f}", "label": args.label,
                "smi_used_mib": smi_used, "smi_free_mib": smi_free,
                "torch_free_mib": torch_free, "torch_total_mib": torch_total,
                "ram_used_mib": ram_u, "ram_avail_mib": ram_a,
                "swap_used_mib": swap_u, "swap_free_mib": swap_f,
            })
            n += 1
            if n % 20 == 0:
                fh.flush()  # survive a hard kill without losing the run

            slept = time.time() - tick
            if slept < period:
                time.sleep(period - slept)

    print(f"[sampler] {n} samples -> {args.out}", file=sys.stderr)
    print(f"[sampler] peak nvidia-smi used: {peak_smi} MiB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
