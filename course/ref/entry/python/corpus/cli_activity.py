"""The `corpus` CLI, Pass 8 verbs: the subprocess activities of CorpusBuild
(data.09; contracts/spec/subprocess-activity.md).

    run --stage <fetch|shard|tokenize> --spec <dir>/spec.json --progress <dir>/progress.jsonl
    cleanup --spec <dir>/spec.json --progress <dir>/progress.jsonl

For `run --stage`, spec.json is the corpus config (the TOML of
formats/corpus-config.schema.json, written by the worker as JSON); relative
source paths and protected sets resolve against TL_ARTIFACTS. Each stage is
one of the pipeline's on-disk boundaries:

    fetch     `run --until fetch`: raw documents and ledger rows
    shard     `run --until shard`: filters, dedup, PII, shards, manifest, ledger check
    tokenize  token streams from the manifest the shard stage wrote

For `cleanup` (CorpusBuild's compensation), spec.json names the build
({"dataset", "version", "tokenizer_id"}): it deletes corpus/<dataset>/<version>/
and tokens/<tokenizer_id>/<dataset>/, and never the raw cache.

Every stage runs under tinyllm.io.activity.run, so the exit codes, DONE.json,
and the work-dir lock are those of dur.09, and spans go to the collector as
`corpus.stage <stage>` children of the activity span.

Entry-point territory (D16): this file is yours.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

STAGES = ("fetch", "shard", "tokenize")


def claims(argv: list[str]) -> bool:
    return bool(argv) and (
        (argv[0] == "run" and any(a == "--stage" or a.startswith("--stage=") for a in argv))
        or argv[0] == "cleanup"
    )


def intercept(argv: list[str]) -> int | None:
    if not claims(argv):
        return None
    from tinyllm.io.activity import run

    if argv[0] == "cleanup":
        return run(cleanup_main, argv[1:])
    p = argparse.ArgumentParser(prog="corpus run", add_help=False)
    p.add_argument("--stage", required=True)
    a, _ = p.parse_known_args(argv[1:])
    return run(lambda act: stage_main(act, a.stage), argv[1:])


def stage_main(act, stage: str) -> list[str]:
    import corpus.__main__ as cli  # noqa: PLC0415  (the CLI's run machinery)

    from tinyllm.io.activity import RetryableError, SpecError
    from tinyllm.io.telemetry import Tracer

    if stage not in STAGES:
        raise SpecError(f"--stage must be one of {', '.join(STAGES)}, got {stage!r}")
    cfg = act.load_spec(lambda c: [f"missing {k!r}" for k in ("dataset", "version", "sources", "shard", "tokenizer") if k not in c])
    art = Path(act.artifacts)
    cdir = art / "corpus" / cfg["dataset"] / cfg["version"]
    tokdir = art / "tokens" / cfg["tokenizer"]["id"] / cfg["dataset"]
    tracer = Tracer.from_env()
    with tracer.span(f"corpus.stage {stage}", {"tl.corpus.stage": stage}) as span:
        try:
            if stage == "tokenize" and (cdir / "_MANIFEST.json").is_file():
                out = cli.tokenize_only(cfg, art, art)
                record_run(art, cfg, out)
            else:
                out = cli.cmd_run(argparse.Namespace(config=cfg, base=art, until=stage, workers=1))
        except cli.Fail as e:
            if e.code == cli.EX_TEMPFAIL:
                raise RetryableError(str(e)) from None
            if e.code == cli.EX_DATAERR:
                raise SpecError(str(e)) from None
            raise
        span.set_attribute("tl.corpus.rows_out", int(out.get("n_docs", 0)))
    tracer.shutdown()
    for k in ("n_docs", "n_shards", "exact_dropped", "near_dropped", "pii_redactions", "train_tokens", "val_tokens"):
        if isinstance(out.get(k), int):
            act.metric(f"corpus.{k}", out[k])
    if stage == "fetch":
        return ["corpus/LEDGER.jsonl"]
    if stage == "shard":
        return [(cdir / "_MANIFEST.json").relative_to(art).as_posix()]
    return [tokdir.relative_to(art).as_posix()]


def record_run(art: Path, cfg: dict, out: dict) -> None:
    """Fold the tokenize stage's counts into corpus/RUN-<dataset>-<version>.json,
    which the shard stage wrote, so the file holds what `corpus run` prints
    (`ctl data build --wait` reads it through the worker)."""
    path = art / "corpus" / f"RUN-{cfg['dataset']}-{cfg['version']}.json"
    run = json.loads(path.read_text()) if path.is_file() else {}
    run.update(out)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run) + "\n")
    os.replace(tmp, path)


def cleanup_main(act) -> list[str]:
    spec = act.load_spec(lambda s: [f"missing {k!r}" for k in ("dataset", "version", "tokenizer_id") if not s.get(k)])
    art = Path(act.artifacts)
    for d in (art / "corpus" / spec["dataset"] / spec["version"], art / "tokens" / spec["tokenizer_id"] / spec["dataset"]):
        if d.exists():
            shutil.rmtree(d)
    return []
