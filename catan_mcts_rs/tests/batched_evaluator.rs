//! Task 10: batched evaluator is REPRODUCIBLE (same inputs -> identical
//! outputs, the replay contract) and FAITHFUL (within float-reassoc bound of
//! the per-state forward). Bit-identity to evaluate_one is NOT expected
//! (batched mean-pooling reassociates ~1e-7) — the batched path is its own
//! canonical reference.
//!
//! Needs a batched .ts: built from the spike net via export_batched. We reuse
//! the production checkpoint's batched export written by the test harness.

use catan_engine::observation::Observation;
use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use std::fs;
use std::path::PathBuf;
use tch::Device;

fn spike_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn obs_from_golden(i: usize) -> Observation {
    let dir = spike_dir();
    let hex = read_f32(&dir.join("g_hex.bin"));
    let vertex = read_f32(&dir.join("g_vertex.bin"));
    let edge = read_f32(&dir.join("g_edge.bin"));
    let scalars = read_f32(&dir.join("g_scalars.bin"));
    let (sh, sv, se, ss) = (19 * 8, 54 * 13, 72 * 6, 59);
    Observation {
        hex_features: hex[i * sh..(i + 1) * sh].to_vec(),
        vertex_features: vertex[i * sv..(i + 1) * sv].to_vec(),
        edge_features: edge[i * se..(i + 1) * se].to_vec(),
        scalars: scalars[i * ss..(i + 1) * ss].to_vec(),
        legal_mask: vec![false; 280],
    }
}

#[test]
fn batched_evaluator_reproducible_and_faithful() {
    // The batched .ts for the spike net (B_MAX=8) is written by
    // mcts_study/scripts/export_spike_batched.py before this test runs.
    let ts = spike_dir().join("wrapper_batched.ts");
    if !ts.exists() {
        eprintln!("skip: {ts:?} missing (run scripts/export_spike_batched.py)");
        return;
    }
    let ev_b = TorchScriptEvaluator::load_batched(ts.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let single = spike_dir().join("wrapper_traced.ts");
    let ev_1 = TorchScriptEvaluator::load(single.to_str().unwrap(), Device::Cpu).unwrap();

    let obss: Vec<Observation> = (0..5).map(obs_from_golden).collect();
    let refs: Vec<&Observation> = obss.iter().collect();

    let out_a = ev_b.evaluate_batch(&refs);
    let out_b = ev_b.evaluate_batch(&refs);
    assert_eq!(out_a.len(), 5);
    // Reproducible: identical bit-for-bit across calls.
    for (a, b) in out_a.iter().zip(out_b.iter()) {
        assert_eq!(a.logits, b.logits, "batched not reproducible (logits)");
        assert_eq!(a.value, b.value, "batched not reproducible (value)");
    }
    // Faithful: within float-reassoc bound of the per-state forward.
    for (i, o) in obss.iter().enumerate() {
        let s = ev_1.evaluate_one(o);
        let dl = out_a[i].logits.iter().zip(s.logits.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let dv = out_a[i].value.iter().zip(s.value.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(dl < 1e-4, "logits diff {dl} too large at {i}");
        assert!(dv < 1e-4, "value diff {dv} too large at {i}");
    }
}
