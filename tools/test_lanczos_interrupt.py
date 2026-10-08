#!/usr/bin/env python3
"""Interrupts that arrive before the check vector is four iterations old.

  python3 tools/test_lanczos_interrupt.py OLD_MSIEVE NEW_MSIEVE WORK_DIR [ROWS]

A stop signal forces an integrity check and a checkpoint, but the check is
only valid against a check vector more than three iterations old. This test
builds its own synthetic job with test_lanczos_packed.py's fixture code (no
job files are touched) and sends SIGTERM the moment Lanczos starts, and the
moment a -ncr restart resumes. OLD_MSIEVE is a build without the fix (it
should fail its forced check); NEW_MSIEVE must defer the stop by up to four
iterations, write a checkpoint, and resume from it to dependencies that are
verified against the original matrix. NEW_MSIEVE must also skip the
square root when -nc3 follows an interrupted solve, and still start it after
a completed one; OLD_MSIEVE is only used for the deferral cases. The signal timing relies on CPU
iterations taking tens of milliseconds: use CPU builds (same VBITS) and the
default 600000 rows. Expect several minutes for the final full solve.
"""
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lanczos_packed import make_job, make_matrix, verify_dependencies

old, new, work = (Path(a).resolve() for a in sys.argv[1:4])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 600000
work.mkdir(parents=True, exist_ok=True)
matrix = work / "synthetic.mat"
if not matrix.exists():
    t = time.time()
    make_job(work)
    make_matrix(matrix, rows=rows, excess=320)
    print(f"made {rows}-row matrix in {time.time() - t:.0f}s", flush=True)


def case(name, from_checkpoint=None):
    prefix = work / name
    for suffix in (".mat", ".log", ".chk", ".bak.chk", ".dep", ".out"):
        Path(str(prefix) + suffix).unlink(missing_ok=True)
    Path(str(prefix) + ".mat").symlink_to(matrix)
    if from_checkpoint:
        shutil.copy(from_checkpoint, str(prefix) + ".chk")
    return prefix


def run(binary, prefix, restart=False, stop_on=None, extra=()):
    log = Path(str(prefix) + ".log")
    cmd = [str(binary), "-s", str(prefix), "-l", str(log), "-t", "1",
           "-i", str(work / "worktodo.ini"), "-nf", str(work / "synthetic.fb")]
    if restart:
        cmd.append("-ncr")
    cmd += ["-nc2", "skip_matbuild=1 dump_interval=10000000", *extra]
    out = open(str(prefix) + ".out", "wb")
    proc = subprocess.Popen(cmd, cwd=work, stdout=out, stderr=out)
    sent = None
    if stop_on:
        # poll the log as tightly as practical and stop on the first sight
        while proc.poll() is None:
            if log.exists() and stop_on in log.read_text(errors="replace"):
                proc.send_signal(signal.SIGTERM)
                sent = time.time()
                break
            time.sleep(0.0005)
    proc.wait(timeout=1800)
    out.close()
    text = log.read_text(errors="replace")
    output = Path(str(prefix) + ".out").read_text(errors="replace")
    halted = re.findall(r"lanczos halted after (\d+) iterations", text)
    restart_at = re.findall(r"restarting at iteration (\d+)", text)
    return {"rc": proc.returncode, "corrupt": "corrupt state" in text + output,
            "halted": int(halted[-1]) if halted else None,
            "restart_at": int(restart_at[-1]) if restart_at else None,
            "signal_sent": sent is not None,
            "chk": Path(str(prefix) + ".chk").exists(),
            "sqrt": "commencing square root" in text,
            "error": "error:" in text + output}


results = []
def check(label, ok, info):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {label}: {info}", flush=True)

# 1. fresh start, stop as Lanczos begins
r = run(old, case("old-fresh"), stop_on="commencing Lanczos iteration")
check("old binary, stop at start: forced check fails (bug reproduced)",
      r["corrupt"] and r["rc"] != 0, r)
r = run(new, case("new-fresh"), stop_on="commencing Lanczos iteration")
check("new binary, stop at start: deferred to iteration 4, clean checkpoint",
      not r["error"] and r["rc"] == 0 and r["chk"] and r["halted"] == 4, r)
chk = work / "early-stop.chk"
shutil.copy(str(work / "new-fresh") + ".chk", chk)

# 2. restart, stop as Lanczos resumes
r = run(old, case("old-restart", chk), restart=True, stop_on="restarting at iteration")
check("old binary, stop right after restart: forced check fails",
      r["corrupt"] and r["rc"] != 0, r)
r = run(new, case("new-restart", chk), restart=True, stop_on="restarting at iteration")
check("new binary, stop right after restart: deferred 4 iterations",
      not r["error"] and r["rc"] == 0 and r["chk"] and r["restart_at"] is not None
      and r["halted"] == r["restart_at"] + 4, r)

# 3. with -nc3 on the command line, the interrupted solve must not go on
#    to the square root (there are no new dependencies to read)
r = run(new, case("new-sqrt"), stop_on="commencing Lanczos iteration",
        extra=["-nc3"])
check("new binary, stop with -nc3: checkpoint, no square root",
      not r["error"] and r["rc"] == 0 and r["chk"] and not r["sqrt"], r)

# 4. the checkpoint written after a deferred stop resumes to a valid solve,
#    and with -nc3 a completed solve still goes on to the square root. The
#    synthetic job has no relations, so the square root itself then fails;
#    only its start is checked
prefix = case("new-finish", str(work / "new-restart") + ".chk")
r = run(new, prefix, restart=True, extra=["-nc3"])
deps_ok = False
if not r["corrupt"] and Path(str(prefix) + ".dep").exists():
    verify_dependencies(Path(str(prefix) + ".mat"), Path(str(prefix) + ".dep"))
    deps_ok = True
check("new binary resumes the deferred checkpoint to verified dependencies, "
      "then starts the square root", deps_ok and r["sqrt"], r)

print("ALL PASS" if all(results) else "SOME FAILED")
sys.exit(0 if all(results) else 1)
