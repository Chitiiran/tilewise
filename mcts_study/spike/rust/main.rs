//! Phase-0 spike (Rust half): load the traced GnnModel wrapper via tch-rs and
//! prove BIT-EXACT parity with the PyTorch reference over 50 fixed states.
//!
//! Spec §5.1: net forward must match to floating-point identity on CPU
//! (max abs diff = 0). A nonzero diff here is a Phase-0 gate failure to
//! investigate, not tolerate.
//!
//! Reads (all f32 LE, fixed shapes; N from g_meta.txt):
//!   g_hex.bin     [N,19,8]    g_vertex.bin [N,54,13]   g_edge.bin [N,72,6]
//!   g_scalars.bin [N,59]      g_value.bin  [N,4]        g_logits.bin [N,280]
//!   wrapper_traced.ts  -- forward(hex, vertex, edge, scalars) -> (value, logits)

use std::fs;
use std::path::Path;
use tch::{CModule, Device, Kind, Tensor};

const N_HEX: i64 = 19;
const F_HEX: i64 = 8;
const N_VERT: i64 = 54;
const F_VERT: i64 = 13;
const N_EDGE: i64 = 72;
const F_EDGE: i64 = 6;
const N_SCALARS: i64 = 59;
const N_VALUE: i64 = 4;
const N_LOGITS: i64 = 280;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    assert!(bytes.len() % 4 == 0, "{:?} not f32-aligned", path);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Slice the i-th example [stride] floats and wrap as a tensor of `shape`.
fn example_tensor(flat: &[f32], i: usize, stride: usize, shape: &[i64]) -> Tensor {
    let start = i * stride;
    let slice = &flat[start..start + stride];
    Tensor::from_slice(slice).reshape(shape)
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".."); // spike/
    let ts_path = dir.join("wrapper_traced.ts");

    let n: usize = fs::read_to_string(dir.join("g_meta.txt"))
        .expect("g_meta.txt")
        .trim()
        .parse()
        .expect("parse N");
    println!("N = {n}");

    let hex = read_f32(&dir.join("g_hex.bin"));
    let vertex = read_f32(&dir.join("g_vertex.bin"));
    let edge = read_f32(&dir.join("g_edge.bin"));
    let scalars = read_f32(&dir.join("g_scalars.bin"));
    let ref_value = read_f32(&dir.join("g_value.bin"));
    let ref_logits = read_f32(&dir.join("g_logits.bin"));

    let mut module = CModule::load_on_device(&ts_path, Device::Cpu)
        .unwrap_or_else(|e| panic!("load {:?}: {}", ts_path, e));
    module.set_eval();

    let s_hex = (N_HEX * F_HEX) as usize;
    let s_vert = (N_VERT * F_VERT) as usize;
    let s_edge = (N_EDGE * F_EDGE) as usize;
    let s_scal = N_SCALARS as usize;

    let mut max_dv = 0.0f32;
    let mut max_dl = 0.0f32;

    for i in 0..n {
        let t_hex = example_tensor(&hex, i, s_hex, &[N_HEX, F_HEX]);
        let t_vert = example_tensor(&vertex, i, s_vert, &[N_VERT, F_VERT]);
        let t_edge = example_tensor(&edge, i, s_edge, &[N_EDGE, F_EDGE]);
        let t_scal = example_tensor(&scalars, i, s_scal, &[N_SCALARS]);

        let out = module
            .forward_is(&[
                tch::IValue::Tensor(t_hex),
                tch::IValue::Tensor(t_vert),
                tch::IValue::Tensor(t_edge),
                tch::IValue::Tensor(t_scal),
            ])
            .expect("forward");
        let (v, logits) = match out {
            tch::IValue::Tuple(mut t) => {
                assert_eq!(t.len(), 2, "expected (value, logits) tuple");
                let logits = t.pop().unwrap();
                let v = t.pop().unwrap();
                match (v, logits) {
                    (tch::IValue::Tensor(v), tch::IValue::Tensor(l)) => (v, l),
                    _ => panic!("tuple elems not tensors"),
                }
            }
            _ => panic!("forward did not return a tuple"),
        };

        let v = v.to_kind(Kind::Float).reshape([N_VALUE]);
        let logits = logits.to_kind(Kind::Float).reshape([N_LOGITS]);
        let v: Vec<f32> = Vec::<f32>::try_from(v).unwrap();
        let logits: Vec<f32> = Vec::<f32>::try_from(logits).unwrap();

        for k in 0..N_VALUE as usize {
            let d = (v[k] - ref_value[i * N_VALUE as usize + k]).abs();
            max_dv = max_dv.max(d);
        }
        for k in 0..N_LOGITS as usize {
            let d = (logits[k] - ref_logits[i * N_LOGITS as usize + k]).abs();
            max_dl = max_dl.max(d);
        }
    }

    println!("Rust-tch vs PyTorch over {n} states: max|dv|={max_dv:.3e} max|dl|={max_dl:.3e}");
    if max_dv == 0.0 && max_dl == 0.0 {
        println!("PHASE-0 NET PARITY: BIT-EXACT. GATE PASSED.");
    } else {
        println!("PHASE-0 NET PARITY: NONZERO DIFF -> investigate (spec §5.1).");
        std::process::exit(1);
    }
}
