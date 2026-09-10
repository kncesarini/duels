#!/usr/bin/env python3
"""Concatenate two or more `feature_dump.rs` matrices (`DVFD` format) into one.

Written for the corpus-generalization experiment: this project's original
`duels-value` weights were trained on a single `mcts-eval`-only corpus, and
the follow-up trains on a **mix** of a new `mcts-value` self-play corpus
(`value_corpus_mv.rs`) and a fresh `mcts-eval` batch. `feature_dump.rs` turns
one `.jsonl` corpus into one `.bin` matrix at a time; this tool is the small
piece of glue that turns several such matrices into the single `--matrix`
`tools/train_value.py` expects, without re-deriving features or touching the
Rust workspace.

    tools/merge_feature_matrices.py \\
        --out arena/corpus/features-mixed.bin \\
        arena/corpus/features-mv.bin arena/corpus/features-eval-insurance.bin

Each input keeps its own row order; rows are concatenated in the order the
inputs are given. `tools/train_value.py` splits by `seed % 10` regardless of
which file a row came from, so as long as the inputs' seed ranges do not
collide (`value_corpus.rs`'s "seed hygiene" note applies here too), mixing
sources this way is exactly as valid as one bigger single-source run — the
split does not care where a row's search came from, only where its game did.

Refuses to merge matrices whose `num_features`/`num_outcomes`/`version` don't
match (mixing features from different `duels_value::features` layouts would
silently corrupt the training input), and writes a `<out>.json` sidecar that
lists the sources and sums their label/game-kind histograms, the same
provenance discipline `feature_dump.rs`'s own sidecar follows.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"DVFD"
HEADER_BYTES = 32


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES or head[:4] != MAGIC:
        raise SystemExit(f"{path} is not a feature_dump matrix")
    version, n_in, n_out = struct.unpack("<III", head[4:16])
    rows, games = struct.unpack("<QQ", head[16:32])
    return version, n_in, n_out, rows, games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="two or more .bin matrices to merge")
    ap.add_argument("--out", required=True, help="the merged .bin matrix to write")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        raise SystemExit("give at least two input matrices to merge")

    headers = [read_header(p) for p in args.inputs]
    version0, n_in0, n_out0, _, _ = headers[0]
    for p, (version, n_in, n_out, _, _) in zip(args.inputs, headers):
        if (version, n_in, n_out) != (version0, n_in0, n_out0):
            raise SystemExit(
                f"{p}: version/num_features/num_outcomes "
                f"{(version, n_in, n_out)} does not match "
                f"{args.inputs[0]}'s {(version0, n_in0, n_out0)}"
            )

    total_rows = sum(h[3] for h in headers)
    total_games = sum(h[4] for h in headers)
    record_bytes = 12 + 4 * n_in0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as w:
        header = (
            MAGIC
            + struct.pack("<III", version0, n_in0, n_out0)
            + struct.pack("<QQ", total_rows, total_games)
        )
        assert len(header) == HEADER_BYTES
        w.write(header)
        for path, (_, _, _, rows, _) in zip(args.inputs, headers):
            with open(path, "rb") as f:
                f.seek(HEADER_BYTES)
                remaining = rows * record_bytes
                while remaining:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        raise SystemExit(f"{path}: truncated (expected {rows} rows)")
                    w.write(chunk)
                    remaining -= len(chunk)

    # Merge sidecars, if present, for a provenance record on the output.
    label_names = None
    label_counts = [0] * n_out0
    games_by_kind = {}
    sources = []
    for path in args.inputs:
        spath = Path(str(path) + ".json")
        entry = {"path": str(path), "rows": None, "games": None}
        if spath.exists():
            sc = json.loads(spath.read_text())
            entry["rows"] = sc.get("rows")
            entry["games"] = sc.get("games")
            entry["corpus"] = sc.get("corpus")
            if label_names is None:
                label_names = sc.get("label_names")
            for i, c in enumerate(sc.get("label_counts", [])):
                if i < len(label_counts):
                    label_counts[i] += c
            for k, v in sc.get("games_by_kind", {}).items():
                games_by_kind[k] = games_by_kind.get(k, 0) + v
        sources.append(entry)

    sidecar = {
        "kind": "duels-value-features-merged",
        "version": 1,
        "generated_by": "tools/merge_feature_matrices.py",
        "sources": sources,
        "num_features": n_in0,
        "num_outcomes": n_out0,
        "record_bytes": record_bytes,
        "games": total_games,
        "rows": total_rows,
        "label_counts": label_counts,
        "label_names": label_names,
        "games_by_kind": games_by_kind,
    }
    Path(str(out) + ".json").write_text(json.dumps(sidecar, indent=2))

    print(f"wrote {total_rows} rows, {total_games} games from {len(args.inputs)} inputs")
    print(f"  {out}")
    print(f"  {out}.json")


if __name__ == "__main__":
    sys.exit(main())
