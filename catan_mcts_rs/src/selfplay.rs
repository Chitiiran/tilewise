//! Self-play game driver — bit-exact port of
//! catan_mcts.async_mcts.play_one_async_game (the self-play path).
//!
//! ONE NpRng per game (seeded by the game seed) drives chance sampling,
//! Dirichlet root noise, AND temperature sampling — exactly as the Python
//! oracle passes a single np.random.default_rng(seed) through. Records moves in
//! the SelfPlayRecorder schema (the Python recorder writes the parquet; Rust
//! returns the structured record).

use crate::evaluator::TorchScriptEvaluator;
use crate::mcts::{best_action, temperature_sample, Mcts, SearchSession, SessionStep, ACTION_SPACE_SIZE};
use crate::rng::NpRng;
use crate::state;
use catan_engine::observation::Observation;
use catan_engine::Engine;

const N_PLAYERS: usize = 4;
const DEFAULT_TEMP_MOVES: usize = 30;
const DEFAULT_DIRICHLET_ALPHA: f64 = 0.8;
const DEFAULT_DIRICHLET_EPS: f64 = 0.25;
const C_PUCT: f64 = 1.4;

#[derive(Clone)]
pub struct RecordedMove {
    pub current_player: i32,
    pub move_index: usize,
    pub legal_mask: Vec<bool>, // length ACTION_SPACE_SIZE
    pub visit_counts: Vec<i32>, // length ACTION_SPACE_SIZE
    pub action_taken: u32,
    pub root_value: f64,
}

pub struct GameRecord {
    pub seed: u64,
    pub terminal: bool,
    pub winner: i32,
    pub final_vp: [i32; N_PLAYERS],
    pub length_in_moves: u32,
    pub action_history: Vec<u32>,
    pub moves: Vec<RecordedMove>,
}

pub struct SelfPlayConfig {
    pub n_sims: u32,
    pub self_play: bool, // true -> Dirichlet + temperature; false -> greedy
    pub vp_target: u8,
    pub bonuses: bool,
    pub max_steps: u32,
    pub temp_moves: usize,
    pub dirichlet_alpha: f64,
    pub dirichlet_eps: f64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            n_sims: 200,
            self_play: false,
            vp_target: 10,
            bonuses: true,
            max_steps: 200_000,
            temp_moves: DEFAULT_TEMP_MOVES,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
            dirichlet_eps: DEFAULT_DIRICHLET_EPS,
        }
    }
}

fn new_engine(seed: u64, vp_target: u8, bonuses: bool) -> Engine {
    if vp_target == 10 && bonuses {
        Engine::new(seed)
    } else {
        Engine::with_rules(seed, vp_target, bonuses)
    }
}

/// Play one self-play game, returning its record. `ev` is the (single) net
/// evaluator. Mirrors play_one_async_game step-for-step.
pub fn play_one_game(ev: &TorchScriptEvaluator, seed: u64, cfg: &SelfPlayConfig) -> GameRecord {
    // eps>0 only for self_play (data generation); arena/eval is greedy.
    let eps = if cfg.self_play { cfg.dirichlet_eps } else { 0.0 };
    let mut rng = NpRng::from_seed(seed);
    let mut engine = new_engine(seed, cfg.vp_target, cfg.bonuses);
    let mut mcts = Mcts::new(ev, C_PUCT, cfg.dirichlet_alpha, eps);

    let mut moves: Vec<RecordedMove> = Vec::new();
    let mut move_index_by_player = [0usize; N_PLAYERS];
    let mut steps = 0u32;

    while !engine.is_terminal() && steps < cfg.max_steps {
        if engine.is_chance_pending() {
            let chosen = state::sample_chance(&engine, &mut rng);
            engine.apply_chance_outcome(chosen);
            steps += 1;
            continue;
        }
        let legal = engine.legal_actions();
        if legal.len() == 1 {
            engine.step(legal[0]);
            steps += 1;
            continue;
        }
        // Search from the current decision node. (Clone so the tree's root
        // owns its engine; the live engine keeps advancing.)
        let visit_counts = mcts.search(engine.clone(), cfg.n_sims, &mut rng);
        let cp = engine.state.current_player as usize;
        let action = if cfg.self_play {
            let tau = if move_index_by_player[cp] < cfg.temp_moves { 1.0 } else { 0.0 };
            temperature_sample(&visit_counts, tau, &mut rng) as u32
        } else {
            best_action(&visit_counts) as u32
        };
        let mut legal_mask = vec![false; ACTION_SPACE_SIZE];
        for &a in &legal {
            legal_mask[a as usize] = true;
        }
        moves.push(RecordedMove {
            current_player: cp as i32,
            move_index: move_index_by_player[cp],
            legal_mask,
            visit_counts: visit_counts.to_vec(),
            action_taken: action,
            root_value: mcts.last_root_value,
        });
        engine.step(action);
        move_index_by_player[cp] += 1;
        steps += 1;
    }

    let terminal = engine.is_terminal();
    let winner = if terminal {
        let rets = state::returns_abs(&engine);
        let max = rets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max > 0.0 {
            rets.iter().position(|&r| r == max).unwrap() as i32
        } else {
            -1
        }
    } else {
        -1
    };
    let stats = engine.stats();
    let mut final_vp = [0i32; N_PLAYERS];
    for p in 0..N_PLAYERS {
        final_vp[p] = stats.players[p].vp_final as i32;
    }

    GameRecord {
        seed,
        terminal,
        winner,
        final_vp,
        length_in_moves: steps,
        action_history: engine.action_history().to_vec(),
        moves,
    }
}

// ---------------------------------------------------------------------------
// Task 10: batched multi-game self-play. Runs seeds.len() games concurrently;
// every game's MCTS leaf eval pending at a given moment is pooled into ONE
// evaluate_batch call (up to b_max), feeding the GPU instead of one leaf at a
// time. Each game consumes its own NpRng in the exact order play_one_game does,
// so a game record is reproducible and equals what play_one_game would produce
// WITH THE SAME (batched) evaluator. Only the batched forward's ~1e-7
// reassociation differs from the B=1 path (the accepted canonical semantics;
// see project_task10_batching_decision_2026_06_18).
// ---------------------------------------------------------------------------

struct Slot {
    seed: u64,
    engine: Engine,
    rng: NpRng,
    eps: f64,
    moves: Vec<RecordedMove>,
    move_index_by_player: [usize; N_PLAYERS],
    steps: u32,
    session: Option<SearchSession>,
    cur_cp: usize,
    cur_legal_mask: Vec<bool>,
    done: bool,
}

impl Slot {
    fn new(seed: u64, cfg: &SelfPlayConfig) -> Self {
        let eps = if cfg.self_play { cfg.dirichlet_eps } else { 0.0 };
        Slot {
            seed,
            engine: new_engine(seed, cfg.vp_target, cfg.bonuses),
            rng: NpRng::from_seed(seed),
            eps,
            moves: Vec::new(),
            move_index_by_player: [0; N_PLAYERS],
            steps: 0,
            session: None,
            cur_cp: 0,
            cur_legal_mask: Vec::new(),
            done: false,
        }
    }

    /// Advance past chance/single-legal moves to the next multi-legal DECISION,
    /// start a SearchSession, and return its first parked leaf. None if the game
    /// reached terminal/step-cap first (sets done). Mirrors play_one_game's loop.
    fn advance_to_search(&mut self, cfg: &SelfPlayConfig) -> Option<Observation> {
        loop {
            if self.engine.is_terminal() || self.steps >= cfg.max_steps {
                self.done = true;
                return None;
            }
            if self.engine.is_chance_pending() {
                let chosen = state::sample_chance(&self.engine, &mut self.rng);
                self.engine.apply_chance_outcome(chosen);
                self.steps += 1;
                continue;
            }
            let legal = self.engine.legal_actions();
            if legal.len() == 1 {
                self.engine.step(legal[0]);
                self.steps += 1;
                continue;
            }
            self.cur_cp = self.engine.state.current_player as usize;
            let mut mask = vec![false; ACTION_SPACE_SIZE];
            for &a in &legal {
                mask[a as usize] = true;
            }
            self.cur_legal_mask = mask;
            let mut sess = SearchSession::new(
                self.engine.clone(), cfg.n_sims, C_PUCT, cfg.dirichlet_alpha, self.eps);
            match sess.pump(&mut self.rng) {
                SessionStep::NeedEval(obs) => {
                    self.session = Some(sess);
                    return Some(obs);
                }
                SessionStep::Done => {
                    self.session = Some(sess);
                    self.finish_move(cfg);
                    continue;
                }
            }
        }
    }

    /// Session complete: pick action, record move, apply. Mirrors play_one_game.
    fn finish_move(&mut self, cfg: &SelfPlayConfig) {
        let sess = self.session.take().expect("finish_move without session");
        let visit_counts = sess.take_visit_counts();
        let root_value = sess.last_root_value;
        let cp = self.cur_cp;
        let action = if cfg.self_play {
            let tau = if self.move_index_by_player[cp] < cfg.temp_moves { 1.0 } else { 0.0 };
            temperature_sample(&visit_counts, tau, &mut self.rng) as u32
        } else {
            best_action(&visit_counts) as u32
        };
        self.moves.push(RecordedMove {
            current_player: cp as i32,
            move_index: self.move_index_by_player[cp],
            legal_mask: std::mem::take(&mut self.cur_legal_mask),
            visit_counts: visit_counts.to_vec(),
            action_taken: action,
            root_value,
        });
        self.engine.step(action);
        self.move_index_by_player[cp] += 1;
        self.steps += 1;
    }

    fn into_record(self) -> GameRecord {
        let terminal = self.engine.is_terminal();
        let winner = if terminal {
            let rets = state::returns_abs(&self.engine);
            let max = rets.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if max > 0.0 { rets.iter().position(|&r| r == max).unwrap() as i32 } else { -1 }
        } else {
            -1
        };
        let stats = self.engine.stats();
        let mut final_vp = [0i32; N_PLAYERS];
        for p in 0..N_PLAYERS {
            final_vp[p] = stats.players[p].vp_final as i32;
        }
        GameRecord {
            seed: self.seed,
            terminal,
            winner,
            final_vp,
            length_in_moves: self.steps,
            action_history: self.engine.action_history().to_vec(),
            moves: self.moves,
        }
    }
}

/// Run all `seeds` games with cross-game leaf batching against ONE batched
/// evaluator. Returns records in the SAME order as `seeds`.
pub fn play_games_batched(
    ev: &TorchScriptEvaluator,
    seeds: &[u64],
    cfg: &SelfPlayConfig,
) -> Vec<GameRecord> {
    let b_max = ev.b_max().expect("play_games_batched needs a batched evaluator");
    let mut slots: Vec<Slot> = seeds.iter().map(|&s| Slot::new(s, cfg)).collect();
    let mut parked: Vec<Option<Observation>> =
        slots.iter_mut().map(|sl| sl.advance_to_search(cfg)).collect();

    loop {
        let active: Vec<usize> = (0..slots.len()).filter(|&i| parked[i].is_some()).collect();
        if active.is_empty() {
            break;
        }
        for chunk in active.chunks(b_max) {
            let obs_refs: Vec<&Observation> =
                chunk.iter().map(|&i| parked[i].as_ref().unwrap()).collect();
            let outs = ev.evaluate_batch(&obs_refs);
            for (pos, &i) in chunk.iter().enumerate() {
                let next = {
                    let sl = &mut slots[i];
                    let sess = sl.session.as_mut().unwrap();
                    sess.provide(&outs[pos], &mut sl.rng)
                };
                match next {
                    SessionStep::NeedEval(obs) => parked[i] = Some(obs),
                    SessionStep::Done => {
                        slots[i].finish_move(cfg);
                        parked[i] = slots[i].advance_to_search(cfg);
                    }
                }
            }
        }
    }

    slots.into_iter().map(Slot::into_record).collect()
}

/// Phase-timing breakdown for the batched scheduler (CPU-side investigation).
pub struct ProfileResult {
    pub total_s: f64,
    pub gpu_s: f64,
    pub provide_s: f64,
    pub advance_s: f64,
    pub n_batches: usize,
    pub total_leaves: usize,
    // Fine split of gpu_s (only populated by profile_batched_timed):
    pub marshal_s: f64,  // host-side tensor build (PARALLELIZABLE)
    pub forward_s: f64,  // bare forward_is + CUDA sync (IRREDUCIBLE serial GPU)
    pub extract_s: f64,  // host-side result copy-out (PARALLELIZABLE)
}

/// Instrumented copy of play_games_batched: attributes wall-clock to GPU
/// forward vs CPU provide (expand/tree/backup) vs game advance. Same logic, so
/// the timings reflect the real hot path.
pub fn profile_batched(
    ev: &TorchScriptEvaluator,
    seeds: &[u64],
    cfg: &SelfPlayConfig,
) -> ProfileResult {
    use std::time::Instant;
    let b_max = ev.b_max().expect("profile_batched needs a batched evaluator");
    let t_all = Instant::now();
    let (mut gpu, mut provide, mut advance) = (0.0f64, 0.0f64, 0.0f64);
    let (mut n_batches, mut total_leaves) = (0usize, 0usize);

    let mut slots: Vec<Slot> = seeds.iter().map(|&s| Slot::new(s, cfg)).collect();
    let t0 = Instant::now();
    let mut parked: Vec<Option<Observation>> =
        slots.iter_mut().map(|sl| sl.advance_to_search(cfg)).collect();
    advance += t0.elapsed().as_secs_f64();

    loop {
        let active: Vec<usize> = (0..slots.len()).filter(|&i| parked[i].is_some()).collect();
        if active.is_empty() {
            break;
        }
        for chunk in active.chunks(b_max) {
            let obs_refs: Vec<&Observation> =
                chunk.iter().map(|&i| parked[i].as_ref().unwrap()).collect();
            let tg = Instant::now();
            let outs = ev.evaluate_batch(&obs_refs);
            gpu += tg.elapsed().as_secs_f64();
            n_batches += 1;
            total_leaves += chunk.len();
            for (pos, &i) in chunk.iter().enumerate() {
                let tp = Instant::now();
                let next = {
                    let sl = &mut slots[i];
                    let sess = sl.session.as_mut().unwrap();
                    sess.provide(&outs[pos], &mut sl.rng)
                };
                provide += tp.elapsed().as_secs_f64();
                match next {
                    SessionStep::NeedEval(obs) => parked[i] = Some(obs),
                    SessionStep::Done => {
                        let ta = Instant::now();
                        slots[i].finish_move(cfg);
                        parked[i] = slots[i].advance_to_search(cfg);
                        advance += ta.elapsed().as_secs_f64();
                    }
                }
            }
        }
    }
    ProfileResult {
        total_s: t_all.elapsed().as_secs_f64(),
        gpu_s: gpu,
        provide_s: provide,
        advance_s: advance,
        n_batches,
        total_leaves,
        marshal_s: 0.0,
        forward_s: 0.0,
        extract_s: 0.0,
    }
}

/// Like profile_batched but splits the GPU phase into marshal / forward /
/// extract via evaluate_batch_timed, to size how much of "GPU time" is the
/// irreducible serial forward vs parallelizable host marshaling.
pub fn profile_batched_timed(
    ev: &TorchScriptEvaluator,
    seeds: &[u64],
    cfg: &SelfPlayConfig,
) -> ProfileResult {
    use std::time::Instant;
    let b_max = ev.b_max().expect("batched");
    let t_all = Instant::now();
    let (mut marshal, mut forward, mut extract, mut provide, mut advance) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut n_batches, mut total_leaves) = (0usize, 0usize);

    let mut slots: Vec<Slot> = seeds.iter().map(|&s| Slot::new(s, cfg)).collect();
    let t0 = Instant::now();
    let mut parked: Vec<Option<Observation>> =
        slots.iter_mut().map(|sl| sl.advance_to_search(cfg)).collect();
    advance += t0.elapsed().as_secs_f64();

    loop {
        let active: Vec<usize> = (0..slots.len()).filter(|&i| parked[i].is_some()).collect();
        if active.is_empty() {
            break;
        }
        for chunk in active.chunks(b_max) {
            let obs_refs: Vec<&Observation> =
                chunk.iter().map(|&i| parked[i].as_ref().unwrap()).collect();
            let (outs, m, f, e) = ev.evaluate_batch_timed(&obs_refs);
            marshal += m;
            forward += f;
            extract += e;
            n_batches += 1;
            total_leaves += chunk.len();
            for (pos, &i) in chunk.iter().enumerate() {
                let tp = Instant::now();
                let next = {
                    let sl = &mut slots[i];
                    sl.session.as_mut().unwrap().provide(&outs[pos], &mut sl.rng)
                };
                provide += tp.elapsed().as_secs_f64();
                match next {
                    SessionStep::NeedEval(obs) => parked[i] = Some(obs),
                    SessionStep::Done => {
                        let ta = Instant::now();
                        slots[i].finish_move(cfg);
                        parked[i] = slots[i].advance_to_search(cfg);
                        advance += ta.elapsed().as_secs_f64();
                    }
                }
            }
        }
    }
    ProfileResult {
        total_s: t_all.elapsed().as_secs_f64(),
        gpu_s: marshal + forward + extract,
        provide_s: provide,
        advance_s: advance,
        n_batches,
        total_leaves,
        marshal_s: marshal,
        forward_s: forward,
        extract_s: extract,
    }
}
