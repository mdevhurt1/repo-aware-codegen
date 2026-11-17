"""Queue the runs to execute unattended. Each run is one budget, so a crash costs little.

Usage: python scripts/run_queue.py <phase> [<phase> ...]

A run whose summary.json exists is skipped, so this is safe to restart. Failures are retried
after a wait, because the likely cause is another process filling the GPU, then skipped.
Nothing runs in parallel, so gen_seconds stays comparable across runs.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SMALL = "deepseek-ai/deepseek-coder-1.3b-base"
LARGE = "deepseek-ai/deepseek-coder-5.7bmqa-base"
REVISION = "8a7cf0c8942cc1aa066bf261839650ac55a2ff79"
FIRST_500 = "data/samples/cff_500_seed0.json"
EXTRA_1500 = "data/samples/cff_extra1500_seed1.json"
RANDOM_500 = "data/samples/crf_500_seed0.json"
BUDGETS = [512, 256, 1024, 2048, 4096]
CONTEXT = ["bm25_long", "bm25_short", "oracle", "oracle_long", "oracle_short"]
LOG = ROOT / "outputs"


def step(name, model, split, sample, methods, budget):
    return {
        "run_name": name,
        "model": {"name": model, "quant": "4bit"},
        "data": {
            "dataset": "tianyang/repobench_python_v1.1",
            "revision": REVISION,
            "split": split,
            "sample": sample,
        },
        "methods": methods,
        "budgets": [budget],
        "generation": {"max_new_tokens": 64},
        "output_dir": "outputs",
    }


def phases():
    first = "cross_file_first"
    return {
        # The 500 sample: `none` is budget-independent, so it runs once, at 512, and doubles as a
        # determinism check against outputs/m1_n512.
        "core": [
            step(f"s500_n{n}", SMALL, first, FIRST_500, (["none"] if n == 512 else []) + CONTEXT, n)
            for n in BUDGETS
        ],
        "signatures": [
            step(f"sig500_n{n}", SMALL, first, FIRST_500, ["signatures"], n) for n in BUDGETS
        ],
        "signatures_extra": [
            step(f"sigextra1500_n{n}", SMALL, first, EXTRA_1500, ["signatures"], n) for n in BUDGETS
        ],
        "random": [
            step(
                "rand500_n512",
                SMALL,
                "cross_file_random",
                RANDOM_500,
                ["none", "bm25_long", "bm25_short", "oracle", "oracle_long", "oracle_short"],
                512,
            )
        ],
        "large": [
            step(
                f"large500_n{n}",
                LARGE,
                first,
                FIRST_500,
                (["none"] if n == 512 else []) + ["bm25_long", "bm25_short", "oracle_long"],
                n,
            )
            for n in BUDGETS
        ],
        "power": [
            step(
                f"extra1500_n{n}",
                SMALL,
                first,
                EXTRA_1500,
                (["none"] if n == 512 else []) + ["bm25_long", "bm25_short", "oracle_long"],
                n,
            )
            for n in BUDGETS
        ],
    }


def make_samples():
    """Fresh samples, disjoint from the frozen 500. cff_500_seed0.json is never touched.

    RepoBench v1.1 has a few examples whose gold_snippet_index is past the end of their context
    (7 in cross_file_first, 8 in cross_file_random). They have no gold snippet, so the oracle
    cannot run on them, and they are excluded before sampling.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from rac.data.repobench import load_examples, sample_ids, save_sample

    def valid(split):
        examples = load_examples(split, REVISION)
        return [e for e in examples if 0 <= e.gold_snippet_index < len(e.context)]

    if not (ROOT / EXTRA_1500).exists():
        frozen = set(json.loads((ROOT / FIRST_500).read_text()))
        rest = sorted(e.example_id for e in valid("cross_file_first") if e.example_id not in frozen)
        save_sample(random.Random(1).sample(rest, 1500), ROOT / EXTRA_1500)
    if not (ROOT / RANDOM_500).exists():
        save_sample(sample_ids(valid("cross_file_random"), 500, 0), ROOT / RANDOM_500)


def say(message):
    line = f"{dt.datetime.now():%H:%M:%S} {message}"
    print(line, flush=True)
    with open(LOG / "progress.log", "a") as f:
        f.write(line + "\n")


def sample_gpu(stop):
    """Record GPU memory over time: the plan reports an 8 GB hardware floor."""
    query = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    with open(LOG / "gpu.csv", "a") as f:
        while not stop.wait(30):
            out = subprocess.run(query, capture_output=True, text=True).stdout.strip()
            apps = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                .stdout.strip()
                .replace("\n", ";")
            )
            f.write(f"{dt.datetime.now():%H:%M:%S},{out},{apps}\n")
            f.flush()


def run_step(config):
    name = config["run_name"]
    if (ROOT / "outputs" / name / "summary.json").exists():
        say(f"skip {name}: already complete")
        return True
    path = ROOT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    for attempt in range(1, 4):
        say(f"start {name} (attempt {attempt})")
        start = time.time()
        with open(LOG / f"{name}.log", "a") as log:
            code = subprocess.run(
                [PY, "scripts/run_one.py", str(path.relative_to(ROOT))],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        if code == 0:
            say(f"done  {name} in {(time.time() - start) / 60:.1f} min")
            return True
        say(f"FAIL  {name} exit {code} after {(time.time() - start) / 60:.1f} min")
        if attempt < 3:
            time.sleep(300)
    return False


def main(names):
    LOG.mkdir(parents=True, exist_ok=True)
    make_samples()
    stop = threading.Event()
    threading.Thread(target=sample_gpu, args=(stop,), daemon=True).start()
    table = phases()
    failed = []
    for phase in names:
        say(f"phase {phase}")
        failed += [c["run_name"] for c in table[phase] if not run_step(c)]
    stop.set()
    say(f"finished {names}; not completed: {failed}")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main(sys.argv[1:]))
