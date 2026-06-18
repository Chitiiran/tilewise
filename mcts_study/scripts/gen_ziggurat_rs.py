"""Parse numpy's ziggurat_constants.h into a Rust constants module.

Preserves every digit verbatim (the float tokens are emitted as-is, so the Rust
f64 literals parse to the identical IEEE-754 values numpy compiled). Writes
catan_mcts_rs/src/ziggurat_tables.rs.
"""
import re
from pathlib import Path

HDR = Path(__file__).resolve().parent / "ziggurat_constants.h"
OUT = (Path(__file__).resolve().parents[2] / "catan_mcts_rs" / "src"
       / "ziggurat_tables.rs")


def parse(text, name, is_hex):
    m = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    toks = [t.strip() for t in m.group(1).split(",") if t.strip()]
    vals = []
    for t in toks:
        t = t.rstrip("UL").rstrip("L").rstrip("U")
        vals.append((t, int(t, 16) if is_hex else None))
    return vals


def emit_u64(name, vals):
    lines = [f"pub static {name}: [u64; {len(vals)}] = ["]
    row = []
    for raw, _ in vals:
        row.append(f"0x{int(raw, 16):016X}")
        if len(row) == 4:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("];")
    return "\n".join(lines)


def emit_f64(name, vals):
    lines = [f"pub static {name}: [f64; {len(vals)}] = ["]
    row = []
    for raw, _ in vals:
        row.append(raw)
        if len(row) == 3:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("];")
    return "\n".join(lines)


def main():
    text = HDR.read_text()
    ki = parse(text, "ki_double", True)
    wi = parse(text, "wi_double", False)
    fi = parse(text, "fi_double", False)
    ke = parse(text, "ke_double", True)
    we = parse(text, "we_double", False)
    fe = parse(text, "fe_double", False)
    for nm, a in [("ki", ki), ("wi", wi), ("fi", fi), ("ke", ke), ("we", we), ("fe", fe)]:
        assert len(a) == 256, f"{nm} has {len(a)} entries"

    parts = [
        "//! NumPy ziggurat constant tables (generated from "
        "ziggurat_constants.h by\n//! mcts_study/scripts/gen_ziggurat_rs.py — "
        "DO NOT EDIT BY HAND).\n//! 256 entries each; values verbatim from "
        "numpy so f64 parse is bit-identical.\n",
        emit_u64("KI_DOUBLE", ki),
        emit_f64("WI_DOUBLE", wi),
        emit_f64("FI_DOUBLE", fi),
        emit_u64("KE_DOUBLE", ke),
        emit_f64("WE_DOUBLE", we),
        emit_f64("FE_DOUBLE", fe),
    ]
    OUT.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
