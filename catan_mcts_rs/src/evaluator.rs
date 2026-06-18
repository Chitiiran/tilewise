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
}

impl TorchScriptEvaluator {
    pub fn load(path: &str, device: Device) -> tch::Result<Self> {
        let mut module = CModule::load_on_device(path, device)?;
        module.set_eval();
        Ok(Self { module, device })
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

    /// Batched interface (Task 4: loops evaluate_one; Task 4b replaces with a
    /// true batched forward). Decisions are identical either way — batching
    /// only changes WHEN forwards happen, not their results.
    pub fn evaluate(&self, batch: &[&Observation]) -> Vec<NetOutput> {
        batch.iter().map(|o| self.evaluate_one(o)).collect()
    }
}
