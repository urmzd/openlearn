"""Course performance budgets relative to machine calibration (DESIGN 5.11).

Calibration: `ol bench --calibrate` builds olcourse/calib/calib.c (a tiled
float32 matmul and a decode-step stand-in) with -O2, runs it three times, and
stores the best of each metric in $OL_CACHE/calibration.json. With
--in-cluster the same program runs as a Job in [deploy].namespace (the
kind VM's CPU) and lands in calibration.in-cluster.json.

A module's budget lives in its registry file:

    [bench]
    lang   = "c"                    # c | python | go | rust
    name   = "matmul_256"           # c/python: course/tests/<ID>/bench/<name>.{c,py};
                                    # go: Benchmark<name> in course/tests/go/<id_>/;
                                    # rust: the #[ignore] test <name> in course/tests/rust/<id_>.rs
    metric = "gflops"               # a key of the last stdout JSON line (go: ns_per_op or a reported unit)
    budget = ">= 0.5 * matmul_gflops"   # <op> <number> [* <calibration metric>]

The bench prints its metrics as one JSON object on its last stdout line
(ol_bench.h does this for C). `perf` milestone steps use the same budgets.
"""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
import time
from pathlib import Path

from . import HarnessError, ctx, ids
from .overlay import LIB_FLAGS, Overlay

CALIB_SRC = Path(__file__).resolve().parent / "calib" / "calib.c"
BUDGET = re.compile(
    r"^\s*(>=|<=|>|<)\s*([0-9.eE+-]+)\s*(?:\*\s*([a-z_][a-z0-9_]*))?\s*$"
)


def calibration_path(in_cluster: bool = False) -> Path:
    return ctx.cache_dir() / (
        "calibration.in-cluster.json" if in_cluster else "calibration.json"
    )


def load_calibration(in_cluster: bool = False) -> dict | None:
    try:
        return json.loads(calibration_path(in_cluster).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def calibrate(runs: int = 3) -> dict:
    exe = ctx.cache_dir() / "calib" / "calib"
    exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cc",
        "-std=c11",
        "-D_POSIX_C_SOURCE=200809L",
        "-O2",
        str(CALIB_SRC),
        "-o",
        str(exe),
    ]
    if platform.system() != "Darwin":
        cmd.append("-lm")
    rc, out = ctx.run(cmd)
    if rc != 0:
        raise HarnessError(f"building the calibration kernel failed:\n{out}")
    samples = []
    for _ in range(runs):
        rc, out = ctx.run([str(exe)], timeout=300)
        obj = _last_json(out) if rc == 0 else None
        if obj is None:
            raise HarnessError(f"the calibration kernel failed:\n{out}")
        samples.append(obj)
    metrics = {k: max(s[k] for s in samples) for k in ("matmul_gflops", "decode_tok_s")}
    doc = {
        "metrics": metrics,
        "runs": samples,
        "host": platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": os.cpu_count(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "where": "host",
    }
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(doc, indent=1) + "\n")
    return doc


JOB_NAME = "ol-calibrate"


def in_cluster_manifest(namespace: str, image: str) -> dict:
    src = CALIB_SRC.read_text()
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": JOB_NAME, "namespace": namespace},
                "data": {"calib.c": src},
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": JOB_NAME,
                    "namespace": namespace,
                    "labels": {"app": JOB_NAME},
                },
                "spec": {
                    "backoffLimit": 0,
                    "ttlSecondsAfterFinished": 600,
                    "template": {
                        "metadata": {"labels": {"app": JOB_NAME}},
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "calibrate",
                                    "image": image,
                                    "command": [
                                        "sh",
                                        "-c",
                                        "cc -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -o /tmp/calib /src/calib.c -lm && for i in 1 2 3; do /tmp/calib; done",
                                    ],
                                    "volumeMounts": [
                                        {"name": "src", "mountPath": "/src"}
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "src", "configMap": {"name": JOB_NAME}}
                            ],
                        },
                    },
                },
            },
        ],
    }


def calibrate_in_cluster(deploy: dict, timeout_s: int = 900) -> dict:
    """The same kernel as a Job in [deploy].namespace, through the drill safety
    gate (a kind- or k3d- context that is current, a namespace that exists)."""
    from . import kube

    k = kube.safety_gate(deploy)
    image = str(deploy.get("calibration_image", "gcc:14"))
    man = in_cluster_manifest(k.namespace, image)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(man, f)
        path = f.name
    try:
        k.run(["delete", "job", JOB_NAME, "--ignore-not-found"])
        k.must(["apply", "-f", path])
        k.must(
            [
                "wait",
                "--for=condition=complete",
                f"job/{JOB_NAME}",
                f"--timeout={timeout_s}s",
            ],
            timeout=timeout_s + 30,
        )
        logs = k.must(["logs", f"job/{JOB_NAME}"])
    finally:
        os.unlink(path)
        k.run(["delete", "job", JOB_NAME, "--ignore-not-found"])
        k.run(["delete", "configmap", JOB_NAME, "--ignore-not-found"])
    samples = []
    for line in logs.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not samples:
        raise HarnessError(f"the calibration Job printed no metrics:\n{logs[-2000:]}")
    metrics = {
        key: max(s[key] for s in samples) for key in ("matmul_gflops", "decode_tok_s")
    }
    doc = {
        "metrics": metrics,
        "runs": samples,
        "context": k.context,
        "namespace": k.namespace,
        "image": image,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "where": "in-cluster",
    }
    calibration_path(True).parent.mkdir(parents=True, exist_ok=True)
    calibration_path(True).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


# ---------------------------------------------------------------------------
# budgets


def check_budget(value: float, budget: str, calib: dict | None) -> tuple[bool, str]:
    m = BUDGET.match(budget)
    if not m:
        raise HarnessError(
            f"budget {budget!r}: want `<op> <number> [* <calibration metric>]`"
        )
    op, num, key = m.group(1), float(m.group(2)), m.group(3)
    bound = num
    if key:
        if calib is None:
            raise HarnessError(
                "this budget is relative to the machine: run `ol bench --calibrate` first"
            )
        metrics = calib.get("metrics", {})
        if key not in metrics:
            raise HarnessError(
                f"budget {budget!r}: calibration has no {key!r} ({', '.join(metrics)})"
            )
        bound = num * float(metrics[key])
    ok = {
        ">=": value >= bound,
        "<=": value <= bound,
        ">": value > bound,
        "<": value < bound,
    }[op]
    return ok, f"{value:.4g} {op} {bound:.4g}" + (f" ({num:g} x {key})" if key else "")


def run_bench(
    ov: Overlay, course: Path, mid: str, spec: dict, timeout: float = 600
) -> tuple[dict | None, str]:
    lang = spec.get("lang", "c")
    name = str(spec.get("name", ""))
    if not name:
        raise HarnessError(f"{mid}: [bench] needs `name`")
    test_dir = ov.test_dir(mid)
    if lang == "c":
        src = test_dir / "bench" / f"{name}.c"
        if not src.is_file():
            raise HarnessError(f"{mid}: no bench program {src}")
        objs, err = ov.c_objects(LIB_FLAGS, "lib")
        if err:
            return None, err
        exe = ov.work / "bench" / name
        exe.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "cc",
            "-std=c11",
            "-D_POSIX_C_SOURCE=200809L",
            "-O2",
            f"-I{ov.contracts / 'c' / 'include'}",
            str(src),
            *map(str, objs),
            "-o",
            str(exe),
        ]
        if platform.system() != "Darwin":
            cmd += ["-lm", "-lpthread"]
        rc, out = ctx.run(cmd)
        if rc != 0:
            return None, f"bench does not compile:\n{out}"
        env = ov.env()
        env.pop("RAYON_NUM_THREADS", None)
        rc, out = ctx.run([str(exe)], cwd=ov.work, env=env, timeout=timeout)
    elif lang == "python":
        src = test_dir / "bench" / f"{name}.py"
        if not src.is_file():
            raise HarnessError(f"{mid}: no bench script {src}")
        prefix, env = ov.py_env()
        ov.work.mkdir(parents=True, exist_ok=True)
        rc, out = ctx.run(
            prefix + ["python", str(src)], cwd=ov.work, env=env, timeout=timeout
        )
    elif lang == "go":
        farm, work = ov.build_go()
        env = ov._go_env(work)
        pkg = f"./{ids.underscore(mid)}/..."
        rc, out = ctx.run(
            [
                "go",
                "test",
                "-run",
                "^$",
                "-bench",
                f"^Benchmark{name}$",
                "-benchtime",
                str(spec.get("benchtime", "1s")),
                pkg,
            ],
            cwd=course / "tests" / "go",
            env=env,
            timeout=timeout,
        )
        if rc == 0:
            return _go_metrics(out, name), out
    elif lang == "rust":
        farm = ov.build_rust([mid])
        env = ov._cargo_env()
        env.pop("RAYON_NUM_THREADS", None)
        rc, out = ctx.run(
            [
                "cargo",
                "test",
                "-q",
                "--release",
                "--manifest-path",
                str(farm / "Cargo.toml"),
                "-p",
                "ol-tests",
                "--test",
                ids.underscore(mid),
                "--",
                "--ignored",
                "--exact",
                name,
                "--nocapture",
            ],
            env=env,
            timeout=max(timeout, 900),
        )
    else:
        raise HarnessError(
            f"{mid}: [bench].lang {lang!r} is not c, python, go, or rust"
        )
    if rc != 0:
        return None, f"bench exited {rc}:\n{ctx.tail(out, 30)}"
    obj = _last_json(out)
    if obj is None:
        return None, f"the bench printed no JSON metrics line:\n{ctx.tail(out, 10)}"
    return obj, out


def _go_metrics(out: str, name: str) -> dict | None:
    for line in out.splitlines():
        if line.startswith(f"Benchmark{name}"):
            parts = line.split()
            vals = {}
            for i in range(2, len(parts) - 1, 2):
                try:
                    vals[parts[i + 1].replace("/", "_per_")] = float(parts[i])
                except ValueError:
                    continue
            return vals
    return None
