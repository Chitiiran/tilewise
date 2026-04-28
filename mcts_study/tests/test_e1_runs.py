from pathlib import Path

from catan_mcts.experiments.e1_winrate_vs_random import main


def test_e1_smoke_run(tmp_path: Path):
    # Smoke ONLY — proves the e1 script writes correctly-shaped parquets.
    # Per move: each `mcts_search` runs `sims` random-policy rollouts to terminal,
    # and a Tier-1 game has ~thousand MCTS-decisions, so wall-clock is dominated
    # by the recording loop. Smoke uses sims=2 / num_games=1 → ~1 minute.
    # Production runs (P3.T7) configure sims/games via cli_main.
    out = main(
        out_root=tmp_path,
        num_games=1,
        sims_per_move_grid=[2],
        seed_base=1000,
    )
    moves = (out / "moves.parquet")
    games = (out / "games.parquet")
    assert moves.exists() and games.exists()

    import pyarrow.parquet as pq
    games_df = pq.read_table(games).to_pandas()
    # 1 budget * 1 game = 1 game row
    assert len(games_df) == 1
