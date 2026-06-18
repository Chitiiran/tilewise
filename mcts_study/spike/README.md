# Phase-0 Spike — Rust-MCTS + TorchScript-GNN gate

**Result: PASSED bit-exact (2026-06-17).** Gate from the design spec
(`docs/superpowers/specs/2026-06-17-rust-mcts-torchscript-gnn-design.md` §5).

## What it proves

The trained PyG `GnnModel` can be served from Rust via `tch-rs` with
floating-point-identical output, **without** a fixed-topology net rewrite.

1. `torch.jit.script(GnnModel)` **fails** — PyG `HeteroConv.forward(*args, **kwargs)`
   varargs are unscriptable (`try_script.py`).
2. `torch.jit.trace` on the raw model **fails** — input is a PyG `HeteroDataBatch`,
   tracer can't infer the container type.
3. A plain-tensor `TensorWrapper` (rebuilds HeteroData internally from the fixed
   edge_index) **traces cleanly**, is bit-exact to eager, and **generalizes** to
   unseen states (`try_trace_wrapper.py`).
4. tch-rs 0.24 loads the traced `.ts` against the pip torch 2.11 wheel and
   reproduces value+logits with **max abs diff = 0.0** over 50 fixed states
   (`rust/main.rs`).

## Files

- `try_script.py` — shows script + raw-trace both fail (diagnostic).
- `try_trace_wrapper.py` — the working traced wrapper + generalization check.
- `export_golden.py` — writes `wrapper_traced.ts` + golden `g_*.bin` (50 states).
- `rust/` — standalone tch-rs crate; loads the `.ts`, asserts bit-exact.
- `probe_*.sh` — env probes (libtorch location/ABI, numpy PCG64 internals).
- `run_rust_spike.sh` — build+run recipe (env vars for the pip-wheel libtorch).

## Reproduce

```bash
wsl.exe -d Ubuntu -u chitii -- bash -lc \
  'source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate && \
   cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study/spike && \
   python export_golden.py && bash run_rust_spike.sh'
```

The golden `.bin`/`.ts` are reused by `catan_mcts_rs/tests/evaluator_parity.rs`
(Phase 4). Implementation plan:
`docs/superpowers/plans/2026-06-17-rust-mcts-torchscript-gnn.md`.
