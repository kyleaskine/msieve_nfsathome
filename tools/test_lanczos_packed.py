#!/usr/bin/env python3
"""Validate packed/streamed Lanczos and old/new checkpoint interoperability.

Run from the repository after building a CUDA binary, for example:
  python3 tools/test_lanczos_packed.py --baseline bench/.../msieve-baseline-v128
Uses its own synthetic matrix, input and polynomial in bench/; no job files
are read or written. It may be invoked from outside a job directory.
The baseline must have the same VBITS as --binary (default: repository msieve).
"""
import argparse
from array import array
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import tempfile


def make_matrix(path, rows=80000, excess=320):
    # The first rows columns form a unit triangular matrix, guaranteeing
    # full row rank; extra columns provide dependencies to recover.
    rng = random.Random(173811)
    with path.open("wb") as out:
        out.write(struct.pack("<III", rows, 16, rows + excess))
        for col in range(rows + excess):
            if col < rows:
                entries = sorted(rng.sample(range(col), min(39, col)) + [col])
            else:
                entries = sorted(rng.sample(range(rows), 40))
            dense = sum(1 << r for r in entries if r < 16)
            sparse = [r for r in entries if r >= 16]
            out.write(struct.pack("<I", len(sparse)))
            out.write(struct.pack("<" + "I" * len(sparse), *sparse))
            out.write(struct.pack("<I", dense))


def verify_dependencies(matrix, deps):
    raw = memoryview(matrix.read_bytes())
    rows, dense, cols = struct.unpack_from("<III", raw)
    values = array("Q")
    values.frombytes(deps.read_bytes())
    assert len(values) == cols and any(values), "missing/nontrivial dependencies"
    sums = [0] * rows
    offset = 12
    for col in range(cols):
        weight, = struct.unpack_from("<I", raw, offset)
        offset += 4
        if values[col]:
            for (row,) in struct.iter_unpack("<I", raw[offset:offset + 4 * weight]):
                sums[row] ^= values[col]
        offset += 4 * weight
        bits, = struct.unpack_from("<I", raw, offset)
        offset += 4
        for r in range(dense):
            if bits & (1 << r):
                sums[r] ^= values[col]
    assert offset == len(raw)
    assert not any(sums), "a dependency does not annihilate the ORIGINAL matrix"


def make_job(directory):
    # Known large primes give a composite without small trial factors. Only
    # linear algebra runs; these simple polynomials share root 2 modulo n.
    n = ((1 << 127) - 1) * ((1 << 521) - 1)
    (directory / "worktodo.ini").write_text(f"{n}\n")
    (directory / "synthetic.fb").write_text(
        f"N {n}\nSKEW 1\nR0 -2\nR1 1\nA0 {n - 8}\nA1 0\nA2 0\nA3 1\n")


def run(binary, prefix, options, restart=False, dump_interval=10000000,
        expect_transfers=None):
    log = Path(str(prefix) + ".log")
    command = [str(binary), "-s", str(prefix), "-l", str(log), "-t", "2",
               "-i", str(prefix.parent / "worktodo.ini"),
               "-nf", str(prefix.parent / "synthetic.fb")]
    if restart:
        command.append("-ncr")
    # The historical transpose planner needs a starting row range >= 20:
    # smaller ranges can stop growing because of its integer 5/4 scaling.
    command += ["-nc2", f"skip_matbuild=1 block_nnz=200000 "
                f"dump_interval={dump_interval} " + options]
    output = prefix.parent / (prefix.name + ".out")
    with output.open("wb") as stream:
        proc = subprocess.Popen(command, cwd=prefix.parent, stdout=stream, stderr=stream)
        try:
            proc.wait(timeout=600)
            assert proc.returncode == 0, f"solver failed: {output}"
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    logged = log.read_text(errors="replace")
    text = output.read_text(errors="replace") + logged
    assert "error:" not in text and "lanczos error" not in text, output
    verify_dependencies(Path(str(prefix) + ".mat"), Path(str(prefix) + ".dep"))
    if expect_transfers is not None:
        summaries = re.findall(
            r"matrix transfers: ([0-9.]+) MiB/iteration, ([0-9.]+) "
            r"copies/iteration \((\d+) measured iterations\)", logged)
        assert len(summaries) == int(expect_transfers), (output, summaries)
        if expect_transfers:
            assert all(float(value) > 0 for value in summaries[0]), summaries
            assert int(summaries[0][2]) > 1, "only a startup sample was reported"
            halted, = re.findall(r"lanczos halted after (\d+) iterations", logged)
            if restart:
                initial, = re.findall(r"restarting at iteration (\d+)", logged)
                products = int(halted) - int(initial)
            else:
                products = int(halted) + 1  # includes initial B*x
            assert int(summaries[0][2]) == products - 3, summaries
    print(f"PASS full solve and independent dependency check: {prefix.name}", flush=True)


def copy_checkpoint(source, destination):
    # All writers have exited. Copy complete, atomically renamed snapshots;
    # never race GPU speed against progress reporting or signal delivery.
    with source.open("rb") as stream:
        rows, dimension, iteration, vbits = struct.unpack("<IIII", stream.read(16))
    assert 0 < dimension < rows and iteration > 0, source
    shutil.copyfile(source, destination)
    print(f"Checkpoint {source.name}: dim={dimension}, iteration={iteration}, "
          f"VBITS={vbits}", flush=True)
    return dimension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--binary", default=str(root / "msieve"))
    parser.add_argument("--work-dir", default=str(root / "bench"))
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    binary, baseline = Path(args.binary).resolve(), Path(args.baseline).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="packed-validation-", dir=work_dir))
    print(f"Validation artifacts: {directory}", flush=True)
    make_job(directory)
    matrix = directory / "synthetic.mat"
    make_matrix(matrix)
    options = "single_copy=1 col24=1 row16=1 host_check=1 stream_frac=0.6 stream_slots=3"

    def case(name):
        prefix = directory / name
        Path(str(prefix) + ".mat").symlink_to(matrix)
        return prefix

    run(binary, case("new-full"), options, expect_transfers=True)

    # A complete 80k-row solve leaves checkpoints near 20k, 40k and 60k.
    # Its backup is the 40k snapshot, independent of how fast the GPU runs.
    # Resuming with a 60k interval produces a later snapshot before completion.
    # Both intervals are multiples of the 10k integrity-check cadence.
    original = case("original-full")
    run(baseline, original, "single_copy=1 stream_frac=0.5 stream_slots=2",
        dump_interval=20000)
    resumed = case("new-restart")
    old_dim = copy_checkpoint(Path(str(original) + ".bak.chk"),
                              Path(str(resumed) + ".chk"))
    assert 39000 <= old_dim < 50000, "unexpected fixture checkpoint cadence"
    run(binary, resumed, options, restart=True, dump_interval=60000,
        expect_transfers=True)
    final = case("original-restart")
    new_dim = copy_checkpoint(Path(str(resumed) + ".chk"), Path(str(final) + ".chk"))
    assert old_dim < new_dim < 70000, "new executable did not advance the checkpoint"
    run(baseline, final, "single_copy=1 stream_frac=0.5 stream_slots=2", restart=True)

    run(binary, case("two-copy"),
        "col24=1 row16=1 host_check=1 stream_frac=0.5 stream_slots=2",
        expect_transfers=True)
    run(binary, case("resident"), "single_copy=1 col24=1 row16=1",
        expect_transfers=False)
    print("PASS all packed Lanczos integration checks", flush=True)


if __name__ == "__main__":
    main()
