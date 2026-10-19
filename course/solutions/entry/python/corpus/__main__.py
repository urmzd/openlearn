"""The reference `corpus` CLI, an entry point: glue that composes the corpus
library (data.01 to data.08) into these verbs:

    corpus run --config C [--until STAGE] [--workers N]
    corpus ledger verify --config C
    corpus datasheet --config C [--out FILE]

Paths. Outputs go under $TL_ARTIFACTS (default ./artifacts): the corpus
directory corpus/ (raw/, LEDGER.jsonl, <dataset>/<version>/ shards and
_MANIFEST.json) and tokens/<tokenizer_id>/<dataset>/. Relative source urls
and protected paths in the config resolve against the config's directory:
local sources are served on a loopback port while data.01's fetch (HTTP
only) runs, and the raw documents' urls are written back as the config gave
them, so the output does not depend on the port.

Final JSON line of `run` (also written to corpus/RUN-<dataset>-<version>.json):
{"dataset", "version", "stage", "n_docs", "n_shards", "filters": {...},
"exact_dropped", "near_dropped", "decontaminated", "pii_redactions",
"train_tokens", "val_tokens", "output_sha256"}. output_sha256 is the sha256
of the lines "<path>\\t<sha256 of the file>\\n", sorted by path, of every file
under corpus/<dataset>/<version>/ and tokens/<tokenizer_id>/<dataset>/
(paths relative to $TL_ARTIFACTS): no time, path prefix, or worker count
enters it.

Exit codes (spec/subprocess-activity.md): 0 ok; 65 a data error (a config
that does not parse, a source whose checksum or license fails, ledger
verification); 75 a retryable fetch failure; 2 a usage error.
"""

from __future__ import annotations

import os
import sys

# Run as a script, Python puts this file's directory (python/corpus/) first on
# sys.path, and corpus/tokenize.py would then shadow the standard library's
# tokenize module (imported by logging, inside asyncio). Replace that entry
# with python/, the root of the corpus and tinyllm packages.
if __package__ in (None, ""):
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import asyncio
import dataclasses
import functools
import hashlib
import http.server
import json
import shutil
import subprocess
import threading
import tomllib
from pathlib import Path
from urllib.parse import quote, urlsplit

STAGES = ("fetch", "filter", "dedup_exact", "dedup_near", "pii", "shard", "tokenize")
EX_DATAERR, EX_TEMPFAIL = 65, 75


class Fail(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def artifacts() -> Path:
    return Path(os.environ.get("TL_ARTIFACTS", "artifacts"))


def load_config(path: Path) -> dict:
    try:
        cfg = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise Fail(EX_DATAERR, f"config {path}: {e}") from None
    for key in ("dataset", "version", "sources", "shard", "tokenizer"):
        if key not in cfg:
            raise Fail(EX_DATAERR, f"config {path}: missing {key!r}")
    return cfg


def local(url: str) -> bool:
    return not urlsplit(url).scheme


class LoopbackFiles:
    """Serve one directory on 127.0.0.1 for the duration of a fetch."""

    def __init__(self, root: Path) -> None:
        handler = functools.partial(_Quiet, directory=str(root))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def __enter__(self) -> "LoopbackFiles":
        threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        ).start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass


def served_paths(base: Path, srcs) -> tuple[Path, dict[str, str]]:
    """The directory the loopback server serves and each local source's URL
    path under it. That is the config's base directory, unless a source is an
    absolute path outside it (`ctl data build` sends absolute paths, as the
    worker's base is its artifacts root); then it is the sources' common
    parent directory."""
    paths = {
        s.id: Path(os.path.normpath(base / s.url)) for s in srcs if local(s.url)
    }
    root = base
    if not all(p.is_relative_to(base) for p in paths.values()):
        root = Path(os.path.commonpath([str(p.parent) for p in paths.values()]))
    return root, {sid: quote(p.relative_to(root).as_posix()) for sid, p in paths.items()}


def counted(name: str, stage, counts: dict):
    """Wrap a Stage so counts[name] ends as documents in minus documents out."""

    def run(docs):
        seen = [0, 0]

        def src():
            for d in docs:
                seen[0] += 1
                yield d

        for d in stage(src()):
            seen[1] += 1
            yield d
        counts[name] = seen[0] - seen[1]

    return run


def tree_sha256(art: Path, dirs: list[Path]) -> str:
    lines = []
    for d in dirs:
        for p in sorted(d.rglob("*")) if d.is_dir() else []:
            if p.is_file():
                lines.append(
                    f"{p.relative_to(art).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
                )
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def cmd_run(a) -> dict:
    if isinstance(a.config, dict):
        # An activity (data.09, cli_activity.py): the config arrives parsed,
        # and relative paths resolve against a.base (TL_ARTIFACTS).
        cfg = a.config
        base = Path(a.base).resolve()
        cfg_bytes = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    else:
        cfg_path = Path(a.config)
        cfg = load_config(cfg_path)
        base = cfg_path.resolve().parent
        cfg_bytes = cfg_path.read_bytes()
    art = artifacts()
    corpus_root = art / "corpus"
    until = a.until
    if until not in STAGES:
        raise Fail(2, f"--until must be one of {', '.join(STAGES)}")
    from corpus.fetch import FetchError, fetch, sources_from_config

    try:
        srcs = sources_from_config(cfg)
    except ValueError as e:
        raise Fail(EX_DATAERR, f"config: {e}") from None
    original = {s.id: s.url for s in srcs}
    root, rel = served_paths(base, srcs)
    with LoopbackFiles(root) as files:
        prefix = {sid: files.base + r for sid, r in rel.items()}
        served = [
            dataclasses.replace(s, url=prefix[s.id]) if s.id in prefix else s
            for s in srcs
        ]
        try:
            manifest = asyncio.run(fetch(served, corpus_root, concurrency=4))
        except FetchError as e:
            raise Fail(EX_TEMPFAIL if e.retryable else EX_DATAERR, str(e)) from None
    if not manifest.ok:
        bad = [f.source.id for f in manifest.entries if f.status == "quarantined"]
        raise Fail(EX_DATAERR, f"checksum mismatch, quarantined: {bad}")
    out = {"dataset": cfg["dataset"], "version": cfg["version"], "stage": until}
    if until == "fetch":
        return out
    from corpus.dedup import exact_dedup
    from corpus.filter import (
        gopher_rules,
        lang_filter,
        length_filter,
        normalize_unicode,
        repetition_filter,
    )
    from corpus.ledger import LedgerError, check, latest, read_ledger, reconcile
    from corpus.minhash import decontaminate, near_dedup
    from corpus.pii import scrub_stage
    from corpus.shard import write_shards
    from corpus.stage import compose, extract
    from corpus.tokenize import tokenize_shards

    def restore_urls(docs):
        for d in docs:
            url = d.meta.get("url", "")
            src_url = original[d.source_id]
            pre = prefix.get(d.source_id)
            if pre and url.startswith(pre):
                d = dataclasses.replace(
                    d,
                    meta={**d.meta, "url": src_url + url[len(pre) :]},
                )
            yield d

    f = cfg.get("filters", {})
    rep = f.get("repetition", {})
    counts: dict[str, int] = {}
    filters = [
        counted(
            "length",
            length_filter(
                int(f.get("min_chars", 50)), int(f.get("max_chars", 100_000))
            ),
            counts,
        ),
        counted(
            "lang",
            lang_filter(
                float(f.get("lang_min_conf", 0.65)), tuple(f.get("langs", ("en",)))
            ),
            counts,
        ),
        counted("gopher", gopher_rules(**dict(f.get("gopher", {}))), counts),
        counted(
            "repetition",
            repetition_filter(int(rep.get("n", 3)), float(rep.get("max_frac", 0.3))),
            counts,
        ),
    ]
    dd = cfg.get("dedup", {})
    num_perm, bands = int(dd.get("num_perm", 128)), int(dd.get("bands", 16))
    threshold = float(dd.get("jaccard_threshold", 0.8))
    dc = cfg.get("decontam", {})
    ngram = int(dc.get("ngram", 13))
    protected = [
        p if Path(p).is_absolute() else base / p for p in dc.get("protected", [])
    ]
    ledger = corpus_root / "LEDGER.jsonl"
    policy = {sid: r["pii_policy"] for sid, r in latest(read_ledger(ledger)).items()}
    stages = [restore_urls, normalize_unicode, *filters]
    if STAGES.index(until) >= STAGES.index("dedup_exact"):
        stages.append(
            counted(
                "dedup_exact",
                functools.partial(
                    exact_dedup, bloom_bytes_per_item=dd.get("bloom_bytes_per_item", 10)
                ),
                counts,
            )
        )
    if STAGES.index(until) >= STAGES.index("dedup_near"):
        near = functools.partial(
            near_dedup,
            num_perm=num_perm,
            bands=bands,
            threshold=threshold,
            k=int(dd.get("shingle", 5)),
            seed=int(dd.get("seed", 0)),
            workers=a.workers,
        )
        stages.append(counted("dedup_near", near, counts))
        stages.append(
            counted(
                "decontam",
                functools.partial(decontaminate, protected=protected, n=ngram),
                counts,
            )
        )
    if STAGES.index(until) >= STAGES.index("pii"):
        stages.append(scrub_stage_with(policy, scrub_stage))
    docs = list(compose(*stages)(extract(manifest)))
    out["filters"] = {
        k: counts.get(k, 0) for k in ("length", "lang", "gopher", "repetition")
    }
    out["exact_dropped"] = counts.get("dedup_exact", 0)
    out["near_dropped"] = counts.get("dedup_near", 0)
    out["decontaminated"] = counts.get("decontam", 0)
    out["pii_redactions"] = sum(int(d.meta.get("pii_redactions", 0)) for d in docs)
    out["n_docs"] = len(docs)
    if STAGES.index(until) < STAGES.index("shard"):
        return out
    sh = cfg["shard"]
    cdir = corpus_root / cfg["dataset"] / cfg["version"]
    m = write_shards(
        docs,
        cdir,
        dataset=cfg["dataset"],
        version=cfg["version"],
        shard_rows=int(sh["shard_rows"]),
        val_permille=int(sh.get("val_permille", 5)),
        row_group_bytes=int(sh.get("row_group_bytes", 64 * 1024 * 1024)),
        filters=out["filters"],
        dedup={
            "exact_dropped": out["exact_dropped"],
            "near_dropped": out["near_dropped"],
            "decontaminated": out["decontaminated"],
            "jaccard_threshold": threshold,
            "num_perm": num_perm,
            "bands": bands,
            "ngram": ngram,
        },
        ledger_ref="corpus/LEDGER.jsonl",
        config_sha256=hashlib.sha256(cfg_bytes).hexdigest(),
    )
    out["n_shards"] = m["n_shards"]
    reconcile(ledger, cdir, filters_applied=["length", "lang", "gopher", "repetition"])
    try:
        check(ledger, cdir)
    except LedgerError as e:
        raise Fail(e.exit_code, "ledger verify failed:\n" + str(e)) from None
    dirs = [cdir]
    if until == "tokenize":
        tk = cfg["tokenizer"]
        tok_json = (
            None
            if tk.get("path") is None
            else (
                base / tk["path"]
                if not Path(tk["path"]).is_absolute()
                else Path(tk["path"])
            )
        )
        tdir = art / "tokens" / tk["id"] / cfg["dataset"]
        tm = tokenize_shards(
            cdir / "_MANIFEST.json", tok_json, tdir, tokenizer_id=tk["id"]
        )
        out["train_tokens"] = sum(
            x["n_tokens"] for x in tm.files if x["split"] == "train"
        )
        out["val_tokens"] = sum(x["n_tokens"] for x in tm.files if x["split"] == "val")
        dirs.append(tdir)
    out["output_sha256"] = tree_sha256(art, dirs)
    run_file = corpus_root / f"RUN-{cfg['dataset']}-{cfg['version']}.json"
    run_file.write_text(json.dumps(out) + "\n")
    return out


def tokenize_only(cfg: dict, art: Path, base: Path) -> dict:
    """The tokenize stage alone, over the manifest the shard stage wrote
    (data.09's tokenize activity: the shards are not rebuilt)."""
    from corpus.tokenize import tokenize_shards

    cdir = art / "corpus" / cfg["dataset"] / cfg["version"]
    tk = cfg["tokenizer"]
    tok_json = None if tk.get("path") is None else (base / tk["path"] if not Path(tk["path"]).is_absolute() else Path(tk["path"]))
    tdir = art / "tokens" / tk["id"] / cfg["dataset"]
    tm = tokenize_shards(cdir / "_MANIFEST.json", tok_json, tdir, tokenizer_id=tk["id"])
    return {
        "dataset": cfg["dataset"], "version": cfg["version"], "stage": "tokenize",
        "train_tokens": sum(x["n_tokens"] for x in tm.files if x["split"] == "train"),
        "val_tokens": sum(x["n_tokens"] for x in tm.files if x["split"] == "val"),
        "output_sha256": tree_sha256(art, [cdir, tdir]),
    }


def scrub_stage_with(policy, scrub_stage):
    def run(docs):
        return scrub_stage(docs, policy=policy)

    return run


def cmd_ledger(a) -> dict:
    cfg = load_config(Path(a.config))
    from corpus.ledger import LedgerError, check

    cdir = artifacts() / "corpus" / cfg["dataset"] / cfg["version"]
    try:
        check(artifacts() / "corpus" / "LEDGER.jsonl", cdir)
    except LedgerError as e:
        raise Fail(e.exit_code, str(e)) from None
    return {"ok": True, "dataset": cfg["dataset"], "version": cfg["version"]}


def cmd_datasheet(a) -> None:
    cfg = load_config(Path(a.config))
    from corpus.ledger import datasheet

    art = artifacts()
    tdir = art / "tokens" / cfg["tokenizer"]["id"] / cfg["dataset"]
    text = datasheet(
        art / "corpus" / "LEDGER.jsonl",
        art / "corpus" / cfg["dataset"] / cfg["version"],
        tdir if tdir.is_dir() else None,
    )
    if a.out:
        Path(a.out).write_text(text)
    sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    # Pass 8 (data.09): run --stage and cleanup, the subprocess activities of
    # CorpusBuild, glue in cli_activity.py (it claims only those forms).
    from corpus.cli_activity import intercept

    code = intercept(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    p = argparse.ArgumentParser(prog="corpus")
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--until", default="tokenize")
    r.add_argument("--workers", type=int, default=1)
    led = sub.add_parser("ledger")
    led.add_argument("action", choices=["verify"])
    led.add_argument("--config", required=True)
    ds = sub.add_parser("datasheet")
    ds.add_argument("--config", required=True)
    ds.add_argument("--out")
    a = p.parse_args(argv)
    try:
        if a.verb == "run":
            print(json.dumps(cmd_run(a)))
        elif a.verb == "ledger":
            print(json.dumps(cmd_ledger(a)))
        else:
            cmd_datasheet(a)
    except Fail as e:
        print(f"corpus: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
