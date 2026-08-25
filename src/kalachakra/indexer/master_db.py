"""Compile all phase outputs into the master SQLite dossier DB (PRD page 1).

SQLite is chosen for the master file: a single, universally-accessible artifact the
frontend can query in milliseconds with no PyTorch and no server. One wide
``tokens`` row per archetype holds every scalar profile; the codebook attribution
breakdown and the ecosystem relation graphs live in indexed side tables. The whole
DB is built in a temp file and atomically renamed into place.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

_TYPE = {int: "INTEGER", float: "REAL", str: "TEXT", bool: "INTEGER"}


def _col_type(values):
    for v in values:
        if v is not None:
            return _TYPE.get(type(v), "REAL")
    return "REAL"


def write_master(cfg, phase1, phase2, phase3, phase4, relations, meta, logger=None):
    """Assemble the master SQLite database atomically."""
    k = cfg.codebook_size
    # merge every domain into one row per token
    merged: dict[int, dict] = {t: {} for t in range(k)}
    for src in (phase1, phase2, phase3, phase4):
        for t, d in src.items():
            row = merged.setdefault(int(t), {})
            for key, val in d.items():
                if key == "attribution":                 # list -> json + side table
                    row["attribution_json"] = json.dumps(val)
                else:
                    row[key] = val

    # union of scalar columns, with a stable, readable order
    cols: list[str] = []
    seen = set()
    for t in range(k):
        for key in merged[t]:
            if key not in seen:
                seen.add(key)
                cols.append(key)
    coltypes = {c: _col_type([merged[t].get(c) for t in range(k)]) for c in cols}

    path = Path(cfg.master_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_db_", suffix=".sqlite")
    os.close(fd)
    try:
        con = sqlite3.connect(tmp)
        cur = con.cursor()
        col_defs = ", ".join(f'"{c}" {coltypes[c]}' for c in cols)
        cur.execute(f"CREATE TABLE tokens (token_id INTEGER PRIMARY KEY"
                    f"{', ' + col_defs if col_defs else ''})")
        placeholders = ", ".join(["?"] * (1 + len(cols)))
        cur.executemany(
            f"INSERT INTO tokens VALUES ({placeholders})",
            [tuple([t] + [merged[t].get(c) for c in cols]) for t in range(k)])

        # attribution side table (token, body_index, weight)
        cur.execute("CREATE TABLE attribution (token_id INTEGER, body_index INTEGER, "
                    "weight REAL)")
        attr_rows = []
        for t in range(k):
            aj = merged[t].get("attribution_json")
            if aj:
                for b, w in enumerate(json.loads(aj)):
                    attr_rows.append((t, b, w))
        cur.executemany("INSERT INTO attribution VALUES (?,?,?)", attr_rows)

        # ecosystem relation graphs
        cur.execute("CREATE TABLE transitions (from_token INTEGER, to_token INTEGER, prob REAL)")
        cur.executemany("INSERT INTO transitions VALUES (?,?,?)", relations.get("transitions", []))
        cur.execute("CREATE TABLE exclusion (token_a INTEGER, token_b INTEGER, corr REAL)")
        cur.executemany("INSERT INTO exclusion VALUES (?,?,?)", relations.get("exclusion", []))
        cur.execute("CREATE TABLE symbiosis (token_id INTEGER, halo_token INTEGER, prob REAL)")
        cur.executemany("INSERT INTO symbiosis VALUES (?,?,?)", relations.get("symbiosis", []))
        cur.execute("CREATE TABLE antipode (token_id INTEGER, antipode_token INTEGER, prob REAL)")
        cur.executemany("INSERT INTO antipode VALUES (?,?,?)", relations.get("antipode", []))

        cur.execute("CREATE TABLE run_meta (key TEXT PRIMARY KEY, value TEXT)")
        cur.executemany("INSERT INTO run_meta VALUES (?,?)",
                        [(str(k2), json.dumps(v)) for k2, v in meta.items()])

        for stmt in (
            "CREATE INDEX idx_attr_token ON attribution(token_id)",
            "CREATE INDEX idx_trans_from ON transitions(from_token)",
            "CREATE INDEX idx_excl_a ON exclusion(token_a)",
            "CREATE INDEX idx_sym_token ON symbiosis(token_id)",
            "CREATE INDEX idx_anti_token ON antipode(token_id)",
        ):
            cur.execute(stmt)
        con.commit()
        con.close()
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if logger:
        logger.info(f"[DB] master dossier written: {path} "
                    f"({len(cols)} profile columns x {k} tokens, "
                    f"{len(attr_rows)} attribution rows)")
    return path
