//! TorchScriptEvaluator — the GNN, in Rust, via tch-rs.
//!
//! Wraps a `tch::CModule` loaded from a TorchScript module exported by
//! `catan_gnn.export_torchscript` (a traced plain-tensor wrapper around the PyG
//! GnnModel — see project_phase0_torchscript_spike_2026_06_17). The ONLY thing
//! that touches libtorch.
//!
//! The traced wrapper's forward signature is:
//!   forward(hex[19,8], vertex[54,13], edge[72,6], scalars[59]) -> (value[1,4], logits[1,280])
//! i.e. VALUE first, LOGITS second (confirmed in the Phase-0 spike).
//!
//! Task 4 (this file): per-state forward, bit-exact parity. True cross-leaf
//! batching (the throughput win) is Task 4b, gated by its own parity test.

use catan_engine::observation::{F_EDGE, F_HEX, F_VERT, N_SCALARS, Observation};
use tch::{CModule, Device, IValue, Kind, Tensor};

const ACTION_SPACE_SIZE: usize = 280;
const N_VALUE: usize = 4;

const N_HEX: i64 = 19;
const N_VERT: i64 = 54;
const N_EDGE: i64 = 72;

/// Policy logits over the full action space + the (ego-relative) 4-player value.
pub struct NetOutput {
    pub logits: Vec<f32>, // length ACTION_SPACE_SIZE (280)
    pub value: [f32; N_VALUE], // ego-relative; MCTS rotates to absolute seat
}

pub struct TorchScriptEvaluator {
    module: CModule,
    device: Device,
    /// Fixed batch width of a batched module (the B_MAX it was traced at), or
    /// None for a B=1 (single-graph) module.
    b_max: Option<usize>,
}

impl TorchScriptEvaluator {
    /// Load a B=1 (single-graph) module.
    pub fn load(path: &str, device: Device) -> tch::Result<Self> {
        let mut module = CModule::load_on_device(path, device)?;
        module.set_eval();
        Ok(Self { module, device, b_max: None })
    }

    /// Load a batched module traced at a fixed `b_max` (from export_batched).
    /// CUDA determinism (use_deterministic_algorithms) must be set at export;
    /// tch inherits the same libtorch global flag via the loaded graph.
    pub fn load_batched(path: &str, device: Device, b_max: usize) -> tch::Result<Self> {
        let mut module = CModule::load_on_device(path, device)?;
        module.set_eval();
        Ok(Self { module, device, b_max: Some(b_max) })
    }

    pub fn b_max(&self) -> Option<usize> {
        self.b_max
    }

    /// One observation -> (logits, value). Value left EGO-relative (rotation
    /// happens in MCTS, matching async_mcts._expand_and_evaluate).
    pub fn evaluate_one(&self, obs: &Observation) -> NetOutput {
        let hex = Tensor::from_slice(&obs.hex_features)
            .reshape([N_HEX, F_HEX as i64])
            .to_device(self.device);
        let vertex = Tensor::from_slice(&obs.vertex_features)
            .reshape([N_VERT, F_VERT as i64])
            .to_device(self.device);
        let edge = Tensor::from_slice(&obs.edge_features)
            .reshape([N_EDGE, F_EDGE as i64])
            .to_device(self.device);
        let scalars = Tensor::from_slice(&obs.scalars)
            .reshape([N_SCALARS as i64])
            .to_device(self.device);

        let out = self
            .module
            .forward_is(&[
                IValue::Tensor(hex),
                IValue::Tensor(vertex),
                IValue::Tensor(edge),
                IValue::Tensor(scalars),
            ])
            .expect("TorchScript forward failed");

        let (value_t, logits_t) = match out {
            IValue::Tuple(mut t) => {
                assert_eq!(t.len(), 2, "expected (value, logits) tuple");
                let logits = t.pop().unwrap();
                let value = t.pop().unwrap();
                match (value, logits) {
                    (IValue::Tensor(v), IValue::Tensor(l)) => (v, l),
                    _ => panic!("tuple elements are not tensors"),
                }
            }
            _ => panic!("forward did not return a tuple"),
        };

        let value_v: Vec<f32> = value_t
            .to_kind(Kind::Float)
            .reshape([N_VALUE as i64])
            .try_into()
            .unwrap();
        let logits: Vec<f32> = logits_t
            .to_kind(Kind::Float)
            .reshape([ACTION_SPACE_SIZE as i64])
            .try_into()
            .unwrap();

        let mut value = [0.0f32; N_VALUE];
        value.copy_from_slice(&value_v);
        NetOutput { logits, value }
    }

    /// Batched forward over <= b_max observations. Pads to b_max with copies of
    /// slot 0 (the batched .ts is traced at a fixed b_max), runs ONE forward,
    /// returns the first `batch.len()` outputs. Requires a batched module.
    ///
    /// This is Task 10: it changes WHEN forwards happen (one GPU call for many
    /// leaves) but, with deterministic kernels + fixed b_max + fixed slot order,
    /// the result for a given (states, slot-order) is reproducible run-to-run.
    pub fn evaluate_batch(&self, batch: &[&Observation]) -> Vec<NetOutput> {
        let b_max = self
            .b_max
            .expect("evaluate_batch requires a batched module (load_batched)");
        let k = batch.len();
        assert!(k >= 1 && k <= b_max, "batch len {k} out of [1,{b_max}]");

        // Concatenate features for b_max graphs (pad with slot 0).
        let mut hex = Vec::with_capacity(b_max * (N_HEX * F_HEX as i64) as usize);
        let mut vertex = Vec::with_capacity(b_max * (N_VERT * F_VERT as i64) as usize);
        let mut edge = Vec::with_capacity(b_max * (N_EDGE * F_EDGE as i64) as usize);
        let mut scalars = Vec::with_capacity(b_max * N_SCALARS);
        for i in 0..b_max {
            let o = batch[if i < k { i } else { 0 }];
            hex.extend_from_slice(&o.hex_features);
            vertex.extend_from_slice(&o.vertex_features);
            edge.extend_from_slice(&o.edge_features);
            scalars.extend_from_slice(&o.scalars);
        }
        let bm = b_max as i64;
        let t_hex = Tensor::from_slice(&hex).reshape([bm * N_HEX, F_HEX as i64]).to_device(self.device);
        let t_vert = Tensor::from_slice(&vertex).reshape([bm * N_VERT, F_VERT as i64]).to_device(self.device);
        let t_edge = Tensor::from_slice(&edge).reshape([bm * N_EDGE, F_EDGE as i64]).to_device(self.device);
        let t_scal = Tensor::from_slice(&scalars).reshape([bm, N_SCALARS as i64]).to_device(self.device);

        let out = self
            .module
            .forward_is(&[
                IValue::Tensor(t_hex),
                IValue::Tensor(t_vert),
                IValue::Tensor(t_edge),
                IValue::Tensor(t_scal),
            ])
            .expect("batched TorchScript forward failed");
        let (value_t, logits_t) = match out {
            IValue::Tuple(mut t) => {
                assert_eq!(t.len(), 2, "expected (value, logits) tuple");
                let logits = t.pop().unwrap();
                let value = t.pop().unwrap();
                match (value, logits) {
                    (IValue::Tensor(v), IValue::Tensor(l)) => (v, l),
                    _ => panic!("tuple elements are not tensors"),
                }
            }
            _ => panic!("forward did not return a tuple"),
        };
        // [b_max, 4] and [b_max, 280] -> flat row-major.
        let value_flat: Vec<f32> = value_t.to_kind(Kind::Float).reshape([bm * N_VALUE as i64]).try_into().unwrap();
        let logits_flat: Vec<f32> = logits_t.to_kind(Kind::Float).reshape([bm * ACTION_SPACE_SIZE as i64]).try_into().unwrap();

        (0..k)
            .map(|i| {
                let mut value = [0.0f32; N_VALUE];
                value.copy_from_slice(&value_flat[i * N_VALUE..(i + 1) * N_VALUE]);
                NetOutput {
                    logits: logits_flat[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE].to_vec(),
                    value,
                }
            })
            .collect()
    }

    /// Convenience: per-state forward (B=1 module). Kept for the correctness
    /// path; the batched scheduler uses evaluate_batch.
    pub fn evaluate(&self, batch: &[&Observation]) -> Vec<NetOutput> {
        batch.iter().map(|o| self.evaluate_one(o)).collect()
    }

    /// Same as evaluate_batch but returns timing (marshal_s, forward_s,
    /// extract_s) with a CUDA sync around the bare forward, so we can see how
    /// much of "GPU time" is the irreducible forward_is vs parallelizable
    /// host-side tensor marshaling. (Investigation only; not the hot path.)
    pub fn evaluate_batch_timed(
        &self,
        batch: &[&Observation],
    ) -> (Vec<NetOutput>, f64, f64, f64) {
        use std::time::Instant;
        let b_max = self.b_max.expect("batched module");
        let k = batch.len();
        let t0 = Instant::now();
        let mut hex = Vec::with_capacity(b_max * (N_HEX * F_HEX as i64) as usize);
        let mut vertex = Vec::with_capacity(b_max * (N_VERT * F_VERT as i64) as usize);
        let mut edge = Vec::with_capacity(b_max * (N_EDGE * F_EDGE as i64) as usize);
        let mut scalars = Vec::with_capacity(b_max * N_SCALARS);
        for i in 0..b_max {
            let o = batch[if i < k { i } else { 0 }];
            hex.extend_from_slice(&o.hex_features);
            vertex.extend_from_slice(&o.vertex_features);
            edge.extend_from_slice(&o.edge_features);
            scalars.extend_from_slice(&o.scalars);
        }
        let bm = b_max as i64;
        let t_hex = Tensor::from_slice(&hex).reshape([bm * N_HEX, F_HEX as i64]).to_device(self.device);
        let t_vert = Tensor::from_slice(&vertex).reshape([bm * N_VERT, F_VERT as i64]).to_device(self.device);
        let t_edge = Tensor::from_slice(&edge).reshape([bm * N_EDGE, F_EDGE as i64]).to_device(self.device);
        let t_scal = Tensor::from_slice(&scalars).reshape([bm, N_SCALARS as i64]).to_device(self.device);
        if self.device.is_cuda() {
            tch::Cuda::synchronize(0);
        }
        let marshal_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let out = self
            .module
            .forward_is(&[
                IValue::Tensor(t_hex),
                IValue::Tensor(t_vert),
                IValue::Tensor(t_edge),
                IValue::Tensor(t_scal),
            ])
            .expect("forward");
        if self.device.is_cuda() {
            tch::Cuda::synchronize(0);
        }
        let forward_s = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let (value_t, logits_t) = match out {
            IValue::Tuple(mut t) => {
                let logits = t.pop().unwrap();
                let value = t.pop().unwrap();
                match (value, logits) {
                    (IValue::Tensor(v), IValue::Tensor(l)) => (v, l),
                    _ => panic!("not tensors"),
                }
            }
            _ => panic!("not a tuple"),
        };
        let value_flat: Vec<f32> = value_t.to_kind(Kind::Float).reshape([bm * N_VALUE as i64]).try_into().unwrap();
        let logits_flat: Vec<f32> = logits_t.to_kind(Kind::Float).reshape([bm * ACTION_SPACE_SIZE as i64]).try_into().unwrap();
        let outs: Vec<NetOutput> = (0..k).map(|i| {
            let mut value = [0.0f32; N_VALUE];
            value.copy_from_slice(&value_flat[i * N_VALUE..(i + 1) * N_VALUE]);
            NetOutput { logits: logits_flat[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE].to_vec(), value }
        }).collect();
        let extract_s = t2.elapsed().as_secs_f64();
        (outs, marshal_s, forward_s, extract_s)
    }
}
