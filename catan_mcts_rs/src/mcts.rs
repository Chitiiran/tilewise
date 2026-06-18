//! MCTS search — bit-exact port of catan_mcts/async_mcts.py.
//!
//! Mirrors the Python oracle exactly so visit-counts and chosen actions match
//! bit-for-bit given the same (seed, net, RNG stream):
//!   - UCB: q + c*prior*sqrt(parent.visits)/(1+child.visits), c=1.4, q=0 unvisited
//!   - select_child: strict `>` so ties keep the FIRST (lowest insertion order)
//!   - chance fast-path: resolve runs of chance nodes via the numpy-replica RNG
//!   - leaf eval: TorchScriptEvaluator; priors = f32 softmax over legal logits
//!   - VALUE ROTATION: GNN value is ego-relative; rotate to absolute seat before
//!     backup (value_abs[(leaf_mover+offset)%4] = value[offset]) — the fix from
//!     project_gnn_value_perspective_bug_2026_05_30. Terminal returns are already
//!     absolute.
//!   - root: expand+evaluate, apply Dirichlet noise, count root 1 visit, then
//!     n_sims-1 sims.
//!   - children created in priors (== legal_actions) order so insertion order
//!     matches Python dict order.

use crate::evaluator::TorchScriptEvaluator;
use crate::rng::NpRng;
use crate::state;
use catan_engine::Engine;

pub const ACTION_SPACE_SIZE: usize = 280;
const N_PLAYERS: usize = 4;

struct Node {
    engine: Engine,
    to_play: i32,
    is_expanded: bool,
    // Insertion-ordered children: (action_id, node). Mirrors Python dict order.
    children: Vec<(u32, Node)>,
    prior: f64,
    visit_count: u64,
    value_sum: f64,
}

impl Node {
    fn new(engine: Engine, prior: f64) -> Self {
        let to_play = state::current_player(&engine);
        Node {
            engine,
            to_play,
            is_expanded: false,
            children: Vec::new(),
            prior,
            visit_count: 0,
            value_sum: 0.0,
        }
    }
}

pub struct Mcts<'e> {
    ev: &'e TorchScriptEvaluator,
    c: f64,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    pub last_root_value: f64,
}

impl<'e> Mcts<'e> {
    pub fn new(ev: &'e TorchScriptEvaluator, c: f64, dirichlet_alpha: f64, dirichlet_eps: f64) -> Self {
        Mcts { ev, c, dirichlet_alpha, dirichlet_eps, last_root_value: 0.0 }
    }

    fn ucb_score(&self, parent_visits: u64, child: &Node) -> f64 {
        let q = if child.visit_count > 0 {
            child.value_sum / child.visit_count as f64
        } else {
            0.0
        };
        let u = self.c * child.prior * (parent_visits as f64).sqrt()
            / (1.0 + child.visit_count as f64);
        q + u
    }

    /// Index of the best child (strict `>`, first wins ties — Python dict order).
    fn select_child_idx(&self, node: &Node) -> usize {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (i, (_, child)) in node.children.iter().enumerate() {
            let s = self.ucb_score(node.visit_count, child);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Resolve chance runs, evaluate the leaf, expand children. Returns the
    /// ABSOLUTE-seat value vector to back up. `rng` drives chance sampling.
    fn expand_and_evaluate(&self, node: &mut Node, rng: &mut NpRng) -> [f32; N_PLAYERS] {
        // Resolve any run of chance nodes (Catan setup chains several).
        while node.engine.is_chance_pending() {
            let chosen = state::sample_chance(&node.engine, rng);
            node.engine.apply_chance_outcome(chosen);
            node.to_play = state::current_player(&node.engine);
        }
        if node.engine.is_terminal() {
            // Terminal returns are already absolute-seat indexed.
            return state::returns_abs(&node.engine);
        }
        // Non-terminal GNN leaf.
        let obs = catan_engine::observation::build_observation(
            &node.engine.state,
            node.engine.state.current_player,
        );
        let out = self.ev.evaluate_one(&obs);
        let leaf_mover = node.engine.state.current_player as usize;

        // Rotate ego-relative value to absolute seat order.
        let mut value_abs = [0.0f32; N_PLAYERS];
        for offset in 0..N_PLAYERS {
            value_abs[(leaf_mover + offset) % N_PLAYERS] = out.value[offset];
        }

        // Priors: f32 softmax over legal logits (matches BatchedGnnEvaluator).
        let mut legal = node.engine.legal_actions();
        // legal_actions order = the engine's natural order; Python passes the
        // same Vec to softmax + zip, so children insert in this order.
        let logits = &out.logits;
        let mut max_logit = f32::NEG_INFINITY;
        for &a in &legal {
            max_logit = max_logit.max(logits[a as usize]);
        }
        let mut exps: Vec<f32> = legal.iter().map(|&a| (logits[a as usize] - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in &mut exps {
            *e /= sum;
        }
        for (i, &a) in legal.iter().enumerate() {
            let mut child_engine = node.engine.clone();
            child_engine.step(a);
            // prior stored as f64 (Python: float(p) widens the f32 prob).
            node.children.push((a, Node::new(child_engine, exps[i] as f64)));
        }
        legal.clear();
        node.is_expanded = true;
        value_abs
    }

    fn apply_root_noise(&self, root: &mut Node, rng: &mut NpRng) {
        if self.dirichlet_eps <= 0.0 || root.children.is_empty() {
            return;
        }
        let k = root.children.len();
        let noise = rng.dirichlet(&vec![self.dirichlet_alpha; k]);
        let eps = self.dirichlet_eps;
        for (i, (_, child)) in root.children.iter_mut().enumerate() {
            child.prior = (1.0 - eps) * child.prior + eps * noise[i];
        }
    }

    /// Backup along `path` (indices into the tree). value_vec is absolute-seat.
    fn backup(root: &mut Node, path: &[usize], value_vec: &[f32; N_PLAYERS]) {
        // Walk the path, updating each node. path[0] is the root (empty step).
        fn rec(node: &mut Node, path: &[usize], value_vec: &[f32; N_PLAYERS]) {
            node.visit_count += 1;
            if node.to_play >= 0 {
                node.value_sum += value_vec[node.to_play as usize] as f64;
            }
            if let Some((&idx, rest)) = path.split_first() {
                rec(&mut node.children[idx].1, rest, value_vec);
            }
        }
        // path includes child indices from root downward (root itself implicit).
        rec(root, path, value_vec);
    }

    /// Run search; returns visit_counts over the full action space.
    pub fn search(
        &mut self,
        root_engine: Engine,
        n_sims: u32,
        rng: &mut NpRng,
    ) -> [i32; ACTION_SPACE_SIZE] {
        let mut root = Node::new(root_engine, 0.0);
        let root_value = self.expand_and_evaluate(&mut root, rng);
        self.apply_root_noise(&mut root, rng);
        root.visit_count += 1;
        if root.to_play >= 0 {
            root.value_sum += root_value[root.to_play as usize] as f64;
        }
        for _ in 0..n_sims.saturating_sub(1) {
            // Descend, collecting the child-index path.
            let mut path: Vec<usize> = Vec::new();
            {
                let mut node: &Node = &root;
                while node.is_expanded
                    && !node.children.is_empty()
                    && !node.engine.is_terminal()
                {
                    let idx = self.select_child_idx(node);
                    path.push(idx);
                    node = &node.children[idx].1;
                }
            }
            // Re-descend mutably to the leaf and expand/evaluate it.
            let value_vec = {
                let mut node: &mut Node = &mut root;
                for &idx in &path {
                    node = &mut node.children[idx].1;
                }
                self.expand_and_evaluate(node, rng)
            };
            Self::backup(&mut root, &path, &value_vec);
        }
        self.last_root_value = if root.visit_count > 0 {
            root.value_sum / root.visit_count as f64
        } else {
            0.0
        };
        let mut out = [0i32; ACTION_SPACE_SIZE];
        for (a, child) in &root.children {
            out[*a as usize] = child.visit_count as i32;
        }
        out
    }
}

/// argmax with ties -> lowest index (matches np.argmax).
pub fn best_action(visit_counts: &[i32; ACTION_SPACE_SIZE]) -> usize {
    let mut best = 0usize;
    let mut best_v = visit_counts[0];
    for (i, &v) in visit_counts.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// AlphaZero temperature move selection from visit counts, matching
/// async_mcts.temperature_sample (tau<=1e-6 -> argmax; else sample N^(1/tau)
/// over visited actions via the numpy-replica choice).
pub fn temperature_sample(
    visit_counts: &[i32; ACTION_SPACE_SIZE],
    tau: f64,
    rng: &mut NpRng,
) -> usize {
    if tau <= 1e-6 {
        return best_action(visit_counts);
    }
    let visited: Vec<i64> = (0..ACTION_SPACE_SIZE as i64)
        .filter(|&a| visit_counts[a as usize] != 0)
        .collect();
    if visited.is_empty() {
        return best_action(visit_counts);
    }
    // weights = N(a)^(1/tau); probs = weights / sum. (f64, matching numpy.)
    let inv_tau = 1.0 / tau;
    let weights: Vec<f64> = visited
        .iter()
        .map(|&a| (visit_counts[a as usize] as f64).powf(inv_tau))
        .collect();
    let wsum: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / wsum).collect();
    rng.choice_i64(&visited, &probs) as usize
}
