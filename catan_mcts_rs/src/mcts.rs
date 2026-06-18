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
        Self::finish_expand(node, &out)
    }

    /// Resolve chance + check terminal WITHOUT calling the net. Returns:
    ///   - Ok(value_abs) when the node is terminal (no net needed), or
    ///   - Err(obs) when it's a non-terminal leaf that needs a net forward.
    /// (The session path uses this to PARK the obs and batch across games.)
    fn prepare_expand(
        node: &mut Node,
        rng: &mut NpRng,
    ) -> Result<[f32; N_PLAYERS], catan_engine::observation::Observation> {
        while node.engine.is_chance_pending() {
            let chosen = state::sample_chance(&node.engine, rng);
            node.engine.apply_chance_outcome(chosen);
            node.to_play = state::current_player(&node.engine);
        }
        if node.engine.is_terminal() {
            return Ok(state::returns_abs(&node.engine));
        }
        Err(catan_engine::observation::build_observation(
            &node.engine.state,
            node.engine.state.current_player,
        ))
    }

    /// Apply a net output to a prepared (non-terminal) leaf: rotate value
    /// ego->absolute, build children in legal order. Returns absolute value.
    /// Identical math to the inline path in expand_and_evaluate.
    fn finish_expand(node: &mut Node, out: &crate::evaluator::NetOutput) -> [f32; N_PLAYERS] {
        let leaf_mover = node.engine.state.current_player as usize;
        let mut value_abs = [0.0f32; N_PLAYERS];
        for offset in 0..N_PLAYERS {
            value_abs[(leaf_mover + offset) % N_PLAYERS] = out.value[offset];
        }
        let legal = node.engine.legal_actions();
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
            node.children.push((a, Node::new(child_engine, exps[i] as f64)));
        }
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

// ---------------------------------------------------------------------------
// Pausable search session (Task 10): drives the SAME algorithm as Mcts::search
// but YIELDS at each leaf net-eval so a driver can batch leaves across many
// concurrent games into one TorchScriptEvaluator::evaluate_batch call.
//
// Determinism: a session consumes its own NpRng in the EXACT order search()
// does (root chance/dirichlet, then per-sim chance), and the prepare/finish
// split is byte-identical math to expand_and_evaluate. So the visit_counts a
// session produces equal what search() would produce *with the same evaluator*
// — the only difference vs the B=1 path is the batched forward's ~1e-7
// reassociation, which is the accepted, reproducible canonical engine.
// ---------------------------------------------------------------------------

/// What a session needs next from the driver.
pub enum SessionStep {
    /// Park this leaf observation; resume via `provide(net_out)`.
    NeedEval(catan_engine::observation::Observation),
    /// Search finished; visit counts ready via `take_visit_counts()`.
    Done,
}

enum Phase {
    /// Root not yet expanded: first pump parks the root leaf.
    RootPending,
    /// Root expand parked; finishing applies noise + counts the root visit.
    RootFinish,
    /// Running sims: `remaining` sims left; a leaf is parked at `path`.
    SimParked { remaining: u32, path: Vec<usize> },
    Done,
}

pub struct SearchSession {
    c: f64,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    n_sims: u32,
    root: Node,
    phase: Phase,
    pub last_root_value: f64,
    visit_counts: Option<[i32; ACTION_SPACE_SIZE]>,
}

impl SearchSession {
    pub fn new(
        root_engine: Engine,
        n_sims: u32,
        c: f64,
        dirichlet_alpha: f64,
        dirichlet_eps: f64,
    ) -> Self {
        SearchSession {
            c,
            dirichlet_alpha,
            dirichlet_eps,
            n_sims,
            root: Node::new(root_engine, 0.0),
            phase: Phase::RootPending,
            last_root_value: 0.0,
            visit_counts: None,
        }
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

    fn descend(&self) -> Vec<usize> {
        let mut path = Vec::new();
        let mut node: &Node = &self.root;
        while node.is_expanded && !node.children.is_empty() && !node.engine.is_terminal() {
            let idx = self.select_child_idx(node);
            path.push(idx);
            node = &node.children[idx].1;
        }
        path
    }

    fn node_at_mut(&mut self, path: &[usize]) -> &mut Node {
        let mut node = &mut self.root;
        for &idx in path {
            node = &mut node.children[idx].1;
        }
        node
    }

    /// First step: prepare the ROOT leaf. Root is virtually always
    /// non-terminal (you don't search a finished game) -> park it, return
    /// NeedEval. Degenerate terminal root -> finish root + run sims (which over
    /// a terminal root immediately complete). After this, drive via provide().
    pub fn pump(&mut self, rng: &mut NpRng) -> SessionStep {
        match &self.phase {
            Phase::RootPending => {}
            _ => panic!("pump() may only be called once, first"),
        }
        match Mcts::prepare_expand(&mut self.root, rng) {
            Err(obs) => {
                self.phase = Phase::RootFinish;
                SessionStep::NeedEval(obs)
            }
            Ok(value) => {
                // Degenerate terminal root: no net needed.
                self.apply_root_noise(rng);
                self.root.visit_count += 1;
                if self.root.to_play >= 0 {
                    self.root.value_sum += value[self.root.to_play as usize] as f64;
                }
                self.phase =
                    Phase::SimParked { remaining: self.n_sims.saturating_sub(1), path: Vec::new() };
                self.start_next_sim(rng).unwrap_or(SessionStep::Done)
            }
        }
    }

    /// Begin the next sim: descend, prepare the leaf. If terminal, back up and
    /// recurse; if it needs a net, park and return NeedEval. Returns None when
    /// all sims are exhausted (caller transitions to Done).
    fn start_next_sim(&mut self, rng: &mut NpRng) -> Option<SessionStep> {
        loop {
            let remaining = match &self.phase {
                Phase::SimParked { remaining, .. } => *remaining,
                _ => 0,
            };
            if remaining == 0 {
                self.phase = Phase::Done;
                self.finalize();
                return Some(SessionStep::Done);
            }
            let path = self.descend();
            let prep = {
                let node = self.node_at_mut(&path);
                Mcts::prepare_expand(node, rng)
            };
            match prep {
                Ok(value) => {
                    // Terminal leaf: back up now, consume one sim, continue.
                    Mcts::backup(&mut self.root, &path, &value);
                    self.phase = Phase::SimParked { remaining: remaining - 1, path: Vec::new() };
                    continue;
                }
                Err(obs) => {
                    self.phase = Phase::SimParked { remaining, path };
                    return Some(SessionStep::NeedEval(obs));
                }
            }
        }
    }

    /// Feed back the net output for the most recently parked leaf, then advance.
    /// Returns the next SessionStep (NeedEval or Done).
    pub fn provide(&mut self, out: &crate::evaluator::NetOutput, rng: &mut NpRng) -> SessionStep {
        match std::mem::replace(&mut self.phase, Phase::Done) {
            Phase::RootFinish => {
                let value = Mcts::finish_expand(&mut self.root, out);
                self.apply_root_noise(rng);
                self.root.visit_count += 1;
                if self.root.to_play >= 0 {
                    self.root.value_sum += value[self.root.to_play as usize] as f64;
                }
                self.phase = Phase::SimParked { remaining: self.n_sims.saturating_sub(1), path: Vec::new() };
                match self.start_next_sim(rng) {
                    Some(s) => s,
                    None => SessionStep::Done,
                }
            }
            Phase::SimParked { remaining, path } => {
                let value = {
                    let node = self.node_at_mut(&path);
                    Mcts::finish_expand(node, out)
                };
                Mcts::backup(&mut self.root, &path, &value);
                self.phase = Phase::SimParked { remaining: remaining - 1, path: Vec::new() };
                match self.start_next_sim(rng) {
                    Some(s) => s,
                    None => SessionStep::Done,
                }
            }
            _ => unreachable!("provide() called when no leaf was parked"),
        }
    }

    fn apply_root_noise(&mut self, rng: &mut NpRng) {
        if self.dirichlet_eps <= 0.0 || self.root.children.is_empty() {
            return;
        }
        let k = self.root.children.len();
        let noise = rng.dirichlet(&vec![self.dirichlet_alpha; k]);
        let eps = self.dirichlet_eps;
        for (i, (_, child)) in self.root.children.iter_mut().enumerate() {
            child.prior = (1.0 - eps) * child.prior + eps * noise[i];
        }
    }

    fn finalize(&mut self) {
        self.last_root_value = if self.root.visit_count > 0 {
            self.root.value_sum / self.root.visit_count as f64
        } else {
            0.0
        };
        let mut out = [0i32; ACTION_SPACE_SIZE];
        for (a, child) in &self.root.children {
            out[*a as usize] = child.visit_count as i32;
        }
        self.visit_counts = Some(out);
    }

    pub fn take_visit_counts(&self) -> [i32; ACTION_SPACE_SIZE] {
        self.visit_counts.expect("session not Done")
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
