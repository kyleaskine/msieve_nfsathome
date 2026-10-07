#!/usr/bin/env python3
"""Check benchmark executable selection using fake programs; no GPU access."""
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BENCH = Path(__file__).resolve().parents[1] / "bench_la.sh"


def executable(path, body):
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


class ExecutableSelection(unittest.TestCase):
    def test_bare_name_uses_checked_file(self):
        with tempfile.TemporaryDirectory(prefix="bench-path-") as tmp:
            root = Path(tmp)
            job = root / "job with spaces"
            job.mkdir()
            commands = root / "commands"
            commands.mkdir()
            for name in ("msieve.dat.mat", "msieve.fb", "worktodo.ini"):
                (job / name).touch()
            # Fake solver exits deliberately. The benchmark should mark it
            # FAIL, but must launch and identify this exact executable.
            local = job / "solver"
            executable(local, "echo LOCAL_EXECUTABLE\nexit 1\n")
            executable(commands / "solver", "echo WRONG_PATH_EXECUTABLE\nexit 1\n")
            executable(commands / "nvidia-smi", "echo 0\n")
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])
            for arg in ("solver", str(local)):
                label = "bare" if arg == "solver" else "absolute"
                subprocess.run(
                    ["bash", str(BENCH), "-b", arg, "-n", label, "-T", "0", "-W", "0"],
                    cwd=job, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=30, check=True)
                directory, = (job / "bench").glob(label + "-*")
                self.assertEqual((directory / "stdout.txt").read_text().strip(),
                                 "LOCAL_EXECUTABLE")
                manifest = (directory / "executable.txt").read_text()
                self.assertEqual(manifest.splitlines()[0], str(local))
                self.assertIn(hashlib.sha256(local.read_bytes()).hexdigest(), manifest)
            with (job / "bench/results.tsv").open(newline="") as stream:
                records = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["binary_path"] == str(local) for row in records))
            self.assertTrue(all(row["status"] == "FAIL" for row in records))


if __name__ == "__main__":
    unittest.main()
