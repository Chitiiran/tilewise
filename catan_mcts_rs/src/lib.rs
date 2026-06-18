//! catan_mcts_rs — Rust MCTS + TorchScript-GNN engine for the Catan AZ loop.
//!
//! Built phase-by-phase (see
//! mcts_study/docs/superpowers/plans/2026-06-17-rust-mcts-torchscript-gnn.md).
//! Every unit is double-verified bit-exact against the Python oracle.

pub mod evaluator;
pub mod rng;
pub mod seedseq;
pub mod ziggurat_tables;
