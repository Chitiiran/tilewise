//! Self-play game driver — bit-exact port of
//! catan_mcts.async_mcts.play_one_async_game (the self-play path).
//!
//! ONE NpRng per game (seeded by the game seed) drives chance sampling,
//! Dirichlet root noise, AND temperature sampling — exactly as the Python
//! oracle passes a single np.random.default_rng(seed) through. Records moves in
//! the SelfPlayRecorder schema (the Python recorder writes the parquet; Rust
//! returns the structured record).

use crate::evaluator::TorchScriptEvaluator;
use crate::mcts::{best_action, temperature_sample, Mcts, ACTION_SPACE_SIZE};
use crate::rng::NpRng;
use crate::state;
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
