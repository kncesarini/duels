#!/usr/bin/env python3
"""Regression test for `merge_feature_matrices.py`'s per-row record size.

This is the test that should have caught the bug the autonomous self-play
loop's first real run surfaced: `record_bytes` was hardcoded to matrix
format **version 1**'s row layout (`seed(u32) + label(u32) + sv(f32) + x`,
`12 + 4*n_in` bytes) even for **version 2** matrices (`value_corpus_mv.rs`'s
format-v2 corpora: `seed(u64) + label(u32) + sv(f32) + ply(u32) + x`,
`20 + 4*n_in` bytes). Merging two version-2 matrices silently read the wrong
byte boundaries per row -- the merge still "succeeded" and its own header's
declared row count happened to match the sum of both inputs' claims, but the
actual written bytes did not hold real, aligned records, which
`tools/train_value.py`'s own row-count sanity check then caught downstream
as "file holds only N rows" instead of at the source.

Builds tiny synthetic version-1 and version-2 matrices directly (the same
`DVFD` header + record layout `feature_dump.rs` and `train_value.py` use),
merges them, and checks the merged file is byte-exact: header row count
times this version's own record size must equal the body's actual length,
and reading each row back must reproduce the inputs' seeds in order. This is
exactly the invariant that broke.

Plain `python3`, no pytest dependency (matching this project's tools/*.py
convention). Run directly:

    tools/test_merge_feature_matrices.py
"""

import os
import struct
import subprocess
import sys
import tempfile

MAGIC = b"DVFD"
HEADER_BYTES = 32


def write_matrix(path, version, n_in, n_out, seeds, ply=None):
    """A minimal, correctly-shaped DVFD matrix: one row per seed, all-zero
    feature vectors (the merge tool never reads feature values, only bytes)."""
    rows = len(seeds)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", version, n_in, n_out))
        f.write(struct.pack("<QQ", rows, rows))  # games == rows here, irrelevant to the bug
        for i, seed in enumerate(seeds):
            if version == 1:
                f.write(struct.pack("<IIf", seed, 0, 0.0))
            elif version == 2:
                f.write(struct.pack("<QIfI", seed, 0, 0.0, ply[i] if ply else 0))
            else:
                raise ValueError(version)
            f.write(struct.pack(f"<{n_in}f", *([0.0] * n_in)))


def read_all_seeds(path, version, n_in):
    """Read every row's seed back, failing loudly (like train_value.py's own
    check) if the header's row count and the body's actual length disagree."""
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        assert head[:4] == MAGIC
        v, n_in_read, n_out = struct.unpack("<III", head[4:16])
        rows, games = struct.unpack("<QQ", head[16:32])
        assert v == version and n_in_read == n_in

        seed_fmt, seed_bytes = ("<Q", 8) if version == 2 else ("<I", 4)
        prefix_bytes = seed_bytes + 4 + 4 + (4 if version == 2 else 0)  # + label + sv (+ ply)
        record_bytes = prefix_bytes + 4 * n_in

        body = f.read()
    if len(body) != rows * record_bytes:
        raise AssertionError(
            f"{path}: header claims {rows} rows ({record_bytes} bytes/row = "
            f"{rows * record_bytes} bytes), body holds {len(body)} bytes -- "
            f"exactly the corruption this test exists to catch"
        )
    seeds = []
    for i in range(rows):
        chunk = body[i * record_bytes : i * record_bytes + seed_bytes]
        seeds.append(struct.unpack(seed_fmt, chunk)[0])
    return seeds


def run_merge(inputs, out):
    script = os.path.join(os.path.dirname(__file__), "merge_feature_matrices.py")
    subprocess.run(
        [sys.executable, script, "--out", out, *inputs], check=True, capture_output=True, text=True
    )


def check_version(version, label):
    n_in, n_out = 13, 4
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.bin")
        b = os.path.join(tmp, "b.bin")
        out = os.path.join(tmp, "merged.bin")
        seeds_a = [100, 101, 102]
        seeds_b = [200, 201]
        write_matrix(a, version, n_in, n_out, seeds_a, ply=[1, 2, 3] if version == 2 else None)
        write_matrix(b, version, n_in, n_out, seeds_b, ply=[4, 5] if version == 2 else None)

        run_merge([a, b], out)
        seeds = read_all_seeds(out, version, n_in)
        expected = seeds_a + seeds_b
        assert seeds == expected, f"{label}: seeds {seeds} != expected {expected}"
    print(f"ok: {label} (record size correctly {'12' if version == 1 else '20'} + 4*n_in)")


def main():
    check_version(1, "version 1 (no ply column, u32 seed)")
    check_version(2, "version 2 (ply column, u64 seed) -- the layout that broke")
    print("all merge_feature_matrices.py record-size checks passed")


if __name__ == "__main__":
    main()
