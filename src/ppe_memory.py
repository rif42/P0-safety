"""
ppe_memory.py
=============
Host-memory probe and safe-settings derivation for YOLO training in a container.

Why this exists
---------------
The first training run of this project died with exit 137 on a Windows/WSL2 box.
The Jupyter log told the real story: the server threw `OSError: [Errno 12]
Cannot allocate memory` on a few-hundred-kilobyte `fp.write()`, two minutes after
a 118-second websocket ping timeout. Nothing hit a Docker memory limit - there
wasn't one. The whole WSL2 VM had been starved by the training process until a
trivial allocation failed, and Linux's OOM killer took the largest RSS process,
which was the training kernel.

The defaults that caused it were sized for a large Linux server:

  workers=8      each forked DataLoader worker copy-on-writes the dataset's label
                 structures, and Python's refcounting turns copy-on-write into
                 real copies. Eight workers is eight copies.
  cache="disk"   safe for RAM in principle, but it decodes to full original
                 resolution before the resize, and the mosaic buffer holds those.
  shm_size=8gb   /dev/shm is a RAM-backed tmpfs. On a VM capped at 50 per cent of
                 host RAM, an 8 GiB shm ceiling is most of the budget.
  no mem_limit   so the container could consume the entire VM, and the OOM killer
                 had to guess what to kill.

GPU VRAM was never the constraint. This module measures what is actually
available and derives settings that fit it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

GIB = 1024 ** 3
# Values above this are cgroup's "no limit" sentinel (or close enough).
_UNLIMITED = 2 ** 62


@dataclass
class MemoryFacts:
    total_gb: float          # RAM the kernel reports
    available_gb: float      # RAM actually free for a new allocation
    swap_gb: float           # swap. Turns an OOM kill into slowness.
    shm_gb: float            # /dev/shm capacity - RAM-backed
    cgroup_limit_gb: Optional[float]   # container limit, None if unlimited
    cgroup_used_gb: Optional[float]
    vram_gb: float
    gpu_count: int

    @property
    def usable_gb(self) -> float:
        """The budget to plan against: the tighter of free RAM and headroom
        left inside any container limit."""
        budget = self.available_gb
        if self.cgroup_limit_gb is not None:
            headroom = self.cgroup_limit_gb - (self.cgroup_used_gb or 0.0)
            budget = min(budget, headroom)
        return max(0.0, budget)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usable_gb"] = round(self.usable_gb, 2)
        return d


def _read_int(path: str) -> Optional[int]:
    try:
        txt = Path(path).read_text().strip()
    except OSError:
        return None
    if txt in ("max", "-1", ""):
        return None
    try:
        v = int(txt)
    except ValueError:
        return None
    return None if v >= _UNLIMITED else v


def _meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                try:
                    out[k] = int(parts[0]) * 1024      # kB -> bytes
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def probe() -> MemoryFacts:
    mi = _meminfo()
    total = mi.get("MemTotal", 0)
    avail = mi.get("MemAvailable", mi.get("MemFree", 0))
    swap = mi.get("SwapTotal", 0)

    try:
        st = os.statvfs("/dev/shm")
        shm = st.f_blocks * st.f_frsize
    except OSError:
        shm = 0

    # cgroup v2 first, then v1
    limit = _read_int("/sys/fs/cgroup/memory.max")
    used = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None:
        limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")

    vram, gpus = 0.0, 0
    try:
        import torch
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            vram = torch.cuda.get_device_properties(0).total_memory / GIB
    except Exception:  # noqa: BLE001
        pass

    return MemoryFacts(
        total_gb=round(total / GIB, 2),
        available_gb=round(avail / GIB, 2),
        swap_gb=round(swap / GIB, 2),
        shm_gb=round(shm / GIB, 2),
        cgroup_limit_gb=round(limit / GIB, 2) if limit else None,
        cgroup_used_gb=round(used / GIB, 2) if used else None,
        vram_gb=round(vram, 2),
        gpu_count=gpus,
    )


# --------------------------------------------------------------------------
# Settings derivation
# --------------------------------------------------------------------------
@dataclass
class Recommendation:
    workers: int
    cache: object            # False | "disk" | "ram"
    batch: int
    imgsz: int
    model_scale: str
    warnings: list
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


def recommend(facts: MemoryFacts, imgsz: int = 640, base_imgsz: int = 640,
              n_images: int = 0) -> Recommendation:
    """Derive training settings that fit the measured machine.

    Two independent budgets, and both have to be satisfied:
      host RAM  -> workers and cache
      GPU VRAM  -> model scale and batch
    """
    warn, notes = [], []
    ram = facts.usable_gb

    # ---- workers -------------------------------------------------------
    # Each forked worker costs roughly a copy of the dataset's label structures
    # plus its prefetch and mosaic buffers. Budget ~1.5 GiB per worker and keep
    # 4 GiB back for the main process, CUDA context and the Jupyter server.
    if ram >= 32:
        workers = 8
    elif ram >= 16:
        workers = 4
    elif ram >= 8:
        workers = 2
    else:
        workers = 0
        notes.append("workers=0 loads data in the main process: slower per epoch, "
                     "but it cannot be killed for being the largest process.")

    # ---- cache ---------------------------------------------------------
    # Never "ram" on a constrained box. "disk" is safe for RAM but decodes at
    # original resolution and needs real disk headroom, so False is the default
    # until the machine has room to spare.
    if ram >= 24:
        cache = "disk"
        notes.append('cache="disk" needs disk headroom of roughly 10-30x the '
                     "JPEG size of the dataset. Check free space before an "
                     "overnight run.")
    else:
        cache = False

    # ---- batch ---------------------------------------------------------
    # Activation memory scales with pixel count, so raising imgsz without
    # lowering batch is the classic silent OOM. This is the coupling the first
    # version of this project got wrong.
    vram = facts.vram_gb
    if   vram >= 40: scale, base_batch = "m", 24
    elif vram >= 22: scale, base_batch = "s", 16
    elif vram >= 11: scale, base_batch = "s", 8
    elif vram > 0:   scale, base_batch = "n", 8
    else:            scale, base_batch = "n", 4

    px_ratio = (base_imgsz / max(imgsz, 1)) ** 2
    batch = max(2, int(base_batch * px_ratio))
    if imgsz != base_imgsz:
        notes.append(f"batch scaled {base_batch} -> {batch} for imgsz {base_imgsz} "
                     f"-> {imgsz} (activation memory goes as pixel count).")
        notes.append("Ultralytics accumulates gradients to nbs=64, so a smaller "
                     "batch does not change the effective batch size or the "
                     "learning-rate schedule - only wall-clock per epoch.")

    # ---- warnings ------------------------------------------------------
    if facts.swap_gb < 2:
        warn.append(f"Swap is {facts.swap_gb:.1f} GiB. With little or no swap, a "
                    "memory spike is an instant kill instead of a slow patch. "
                    "Give the VM 8-16 GiB of swap.")
    if facts.shm_gb > max(2.0, ram * 0.35):
        warn.append(f"/dev/shm is {facts.shm_gb:.1f} GiB against {ram:.1f} GiB "
                    "usable RAM. It is a RAM-backed tmpfs, so that ceiling "
                    f"competes with training. Size it near {max(2, int(ram*0.25))} GiB.")
    if facts.cgroup_limit_gb is None:
        warn.append("No container memory limit is set, so this container can "
                    "consume the whole VM and the OOM killer picks its own "
                    "victim. Set mem_limit so failures are attributable.")
    if ram < 8:
        warn.append(f"Only {ram:.1f} GiB usable. Training will be tight whatever "
                    "the settings - raise the VM ceiling before tuning anything else.")
    if facts.gpu_count == 0:
        warn.append("No CUDA device visible. Training on CPU is a smoke test only.")

    return Recommendation(workers=workers, cache=cache, batch=batch, imgsz=imgsz,
                          model_scale=scale, warnings=warn, notes=notes)


def report(facts: MemoryFacts, rec: Optional[Recommendation] = None) -> str:
    L = [
        "Memory",
        f"  RAM total       {facts.total_gb:>8.1f} GiB",
        f"  RAM available   {facts.available_gb:>8.1f} GiB",
        f"  swap            {facts.swap_gb:>8.1f} GiB",
        f"  /dev/shm        {facts.shm_gb:>8.1f} GiB   (RAM-backed)",
    ]
    if facts.cgroup_limit_gb is not None:
        L.append(f"  container limit {facts.cgroup_limit_gb:>8.1f} GiB   "
                 f"(using {facts.cgroup_used_gb or 0:.1f})")
    else:
        L.append("  container limit      none   (can consume the whole VM)")
    L.append(f"  usable budget   {facts.usable_gb:>8.1f} GiB")
    L.append(f"  GPU             {facts.gpu_count} x {facts.vram_gb:.1f} GiB VRAM")

    if rec:
        L += ["", "Derived settings",
              f"  model    yolo26{rec.model_scale}.pt",
              f"  imgsz    {rec.imgsz}",
              f"  batch    {rec.batch}",
              f"  workers  {rec.workers}",
              f"  cache    {rec.cache!r}"]
        if rec.notes:
            L += ["", "Notes"] + [f"  - {n}" for n in rec.notes]
        if rec.warnings:
            L += ["", "WARNINGS"] + [f"  ! {w}" for w in rec.warnings]
    return "\n".join(L)


def preflight(facts: MemoryFacts, rec: Recommendation, min_ram_gb: float = 6.0
              ) -> tuple:
    """Go / no-go before committing to a long run. Returns (ok, message)."""
    if facts.usable_gb < min_ram_gb:
        return False, (f"STOP: {facts.usable_gb:.1f} GiB usable RAM is below the "
                       f"{min_ram_gb:.0f} GiB floor. Raise the VM ceiling "
                       "(.wslconfig on Windows) before starting a run that will "
                       "take hours to fail.")
    if facts.gpu_count == 0:
        return True, ("PROCEED WITH CAUTION: no GPU, so this is a smoke test. "
                      "Use a small --limit and few epochs.")
    if facts.swap_gb < 2 and facts.usable_gb < 16:
        return True, ("PROCEED WITH CAUTION: little RAM and almost no swap. A "
                      "transient spike will kill the run outright. Adding swap "
                      "costs nothing and converts a kill into a slow patch.")
    return True, "OK to proceed."


if __name__ == "__main__":
    f = probe()
    print(report(f, recommend(f, imgsz=960)))
    ok, msg = preflight(f, recommend(f, imgsz=960))
    print(f"\npreflight: {'PASS' if ok else 'FAIL'} - {msg}")
