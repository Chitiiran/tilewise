//! Golden-parity: TorchScriptEvaluator output == PyTorch reference, bit-exact
//! (spec §5.1, max abs diff 0.0 on CPU) over the 50 fixed spike states.
//!
//! Reuses mcts_study/spike/{wrapper_traced.ts, g_*.bin, g_meta.txt} — the same
//! artifacts the Phase-0 spike validated. This is the production-API version of
//! that check (through TorchScriptEvaluator::evaluate rather than raw tch).

use catan_engine::observation::Observation;
use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use std::fs;
use std::path::PathBuf;
use tch::Device;

fn spike_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mcts_study")
        .join("spike")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn evaluator_matches_pytorch_bit_exact() {
    let dir = spike_dir();
    let n: usize = fs::read_to_string(dir.join("g_meta.txt"))
        .expect("g_meta.txt (run spike/export_golden.py)")
        .trim()
        .parse()
        .unwrap();

    let hex = read_f32(&dir.join("g_hex.bin")); // [n,19,8]
    let vertex = read_f32(&dir.join("g_vertex.bin")); // [n,54,13]
    let edge = read_f32(&dir.join("g_edge.bin")); // [n,72,6]
    let scalars = read_f32(&dir.join("g_scalars.bin")); // [n,59]
    let ref_value = read_f32(&dir.join("g_value.bin")); // [n,4]
    let ref_logits = read_f32(&dir.join("g_logits.bin")); // [n,280]

    let s_hex = 19 * 8;
    let s_vert = 54 * 13;
    let s_edge = 72 * 6;
    let s_scal = 59;

    let ev = TorchScriptEvaluator::load(
        dir.join("wrapper_traced.ts").to_str().unwrap(),
        Device::Cpu,
    )
    .expect("load wrapper_traced.ts");

    let mut max_dv = 0.0f32;
    let mut max_dl = 0.0f32;
    for i in 0..n {
        let obs = Observation {
            hex_features: hex[i * s_hex..(i + 1) * s_hex].to_vec(),
            vertex_features: vertex[i * s_vert..(i + 1) * s_vert].to_vec(),
            edge_features: edge[i * s_edge..(i + 1) * s_edge].to_vec(),
            scalars: scalars[i * s_scal..(i + 1) * s_scal].to_vec(),
            legal_mask: vec![false; 280], // unused by the net forward
        };
        let out = ev.evaluate_one(&obs);
        for k in 0..4 {
            max_dv = max_dv.max((out.value[k] - ref_value[i * 4 + k]).abs());
        }
        for k in 0..280 {
            max_dl = max_dl.max((out.logits[k] - ref_logits[i * 280 + k]).abs());
        }
    }
    assert_eq!(max_dv, 0.0, "value not bit-exact (max {max_dv})");
    assert_eq!(max_dl, 0.0, "logits not bit-exact (max {max_dl})");
}
