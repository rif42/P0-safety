"""Tests for ppe_memory using synthetic MemoryFacts - no real machine involved."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ppe_memory import MemoryFacts, recommend, preflight, report

fails = []
def check(label, got, want, extra=""):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {label:<54} got={got}{'  ' + extra if extra else ''}")
    if not ok: fails.append(f"{label} (got {got!r}, want {want!r})")

def F(total=32, avail=24, swap=8, shm=4, limit=None, used=None, vram=24, gpus=1):
    return MemoryFacts(total, avail, swap, shm, limit, used, vram, gpus)

# ---- usable budget -------------------------------------------------------
check("usable = available when no cgroup limit", F(avail=24).usable_gb, 24)
check("usable = cgroup headroom when that is tighter",
      F(avail=24, limit=10, used=2).usable_gb, 8)
check("usable = available when cgroup is roomier",
      F(avail=6, limit=64, used=0).usable_gb, 6)
check("usable never negative", F(avail=24, limit=4, used=9).usable_gb, 0.0)

# ---- workers tiers -------------------------------------------------------
check("32+ GiB -> 8 workers", recommend(F(avail=40)).workers, 8)
check("16-32 GiB -> 4 workers", recommend(F(avail=20)).workers, 4)
check("8-16 GiB -> 2 workers", recommend(F(avail=10)).workers, 2)
check("<8 GiB -> 0 workers (main process)", recommend(F(avail=7)).workers, 0)
check("David's box: 7.2 usable -> 0 workers, not 8",
      recommend(F(avail=7.2)).workers, 0, "<- the setting that killed the run")

# ---- cache ---------------------------------------------------------------
check("roomy box -> disk cache", recommend(F(avail=40)).cache, "disk")
check("constrained box -> no cache", recommend(F(avail=10)).cache, False)
check("cache is never 'ram'",
      any(recommend(F(avail=a)).cache == "ram" for a in (8, 16, 32, 64, 256)), False)

# ---- batch / imgsz coupling (the latent bug) ------------------------------
r640 = recommend(F(vram=16), imgsz=640)
r960 = recommend(F(vram=16), imgsz=960)
r1120 = recommend(F(vram=16), imgsz=1120)
check("imgsz 640 keeps base batch", r640.batch, 8)
check("imgsz 960 lowers batch ~2.25x", r960.batch, 3, f"8 * (640/960)^2 = 3.55 -> 3")
check("imgsz 1120 lowers batch further", r1120.batch, 2)
check("batch is monotone non-increasing in imgsz",
      r640.batch >= r960.batch >= r1120.batch, True)
check("batch never drops below 2", recommend(F(vram=4), imgsz=1600).batch >= 2, True)
check("imgsz change is explained in notes",
      any("batch scaled" in n for n in r960.notes), True)
check("nbs accumulation is explained",
      any("nbs=64" in n for n in r960.notes), True)

# ---- model scale from VRAM ----------------------------------------------
check("40+ GiB VRAM -> m", recommend(F(vram=48)).model_scale, "m")
check("16 GiB VRAM -> s", recommend(F(vram=16)).model_scale, "s")
check("8 GiB VRAM -> n", recommend(F(vram=8)).model_scale, "n")
check("no GPU -> n", recommend(F(vram=0, gpus=0)).model_scale, "n")

# ---- warnings ------------------------------------------------------------
def warns(f, **kw):
    return " ".join(recommend(f, **kw).warnings).lower()

check("low swap warns", "swap" in warns(F(swap=0)), True)
check("adequate swap does not warn", "swap is" in warns(F(swap=16)), False)
check("oversized shm warns", "/dev/shm" in warns(F(avail=8, shm=8)), True)
check("proportionate shm does not warn", "/dev/shm" in warns(F(avail=32, shm=4)), False)
check("missing container limit warns",
      "container memory limit" in warns(F(limit=None)), True)
check("present container limit does not warn",
      "container memory limit" in warns(F(limit=24, used=1)), False)
check("no GPU warns", "cuda" in warns(F(vram=0, gpus=0)), True)

# David's actual configuration, reconstructed
david = F(total=8, avail=7.2, swap=0, shm=8, limit=None, vram=16, gpus=1)
w = warns(david)
check("David's box trips swap+shm+limit+ram warnings",
      all(k in w for k in ("swap", "/dev/shm", "container memory limit", "usable")), True,
      f"{len(recommend(david).warnings)} warnings")

# ---- preflight -----------------------------------------------------------
ok, msg = preflight(F(avail=3), recommend(F(avail=3)))
check("below RAM floor -> no-go", ok, False)
check("no-go message names the fix", ".wslconfig" in msg, True)
ok, msg = preflight(F(avail=24, swap=8), recommend(F(avail=24)))
check("healthy box -> go", (ok, msg), (True, "OK to proceed."))
ok, msg = preflight(F(avail=10, swap=0), recommend(F(avail=10)))
check("tight box + no swap -> go with caution", ok and "CAUTION" in msg, True)

# ---- report renders without a real machine -------------------------------
txt = report(david, recommend(david, imgsz=960))
check("report includes shm line", "/dev/shm" in txt, True)
check("report includes derived batch", "batch" in txt, True)

print("\n" + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES:"))
for f_ in fails: print("  -", f_)
sys.exit(1 if fails else 0)
