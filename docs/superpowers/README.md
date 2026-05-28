# Superpowers docs — how to navigate

This directory holds the project's design specs, implementation plans, and execution journals. All artifacts are markdown, organized by document type and date.

## Navigation

- **Start with [INDEX.md](INDEX.md)** — chronological + topical index across all four project phases, with cumulative-best decision history and a cross-reference table for the four 1200-game tournaments.
- **By document type:**
  - [`specs/`](specs/) — design specs, one per major decision
  - [`plans/`](plans/) — implementation plans, written before each task block
  - [`journals/`](journals/) — execution journals, written during/after each task
  - [`journals/figures/`](journals/figures/) — PNG figures referenced by journals; see [figures/README.md](journals/figures/README.md) for the figure-to-journal map
  - [`reference/`](reference/) — static reference material (board topology tables, etc.)

## Conventions

- Filenames: `YYYY-MM-DD-<kebab-slug>.md`
- Plans are written **before** code; journals are written **during/after**.
- Specs are written when a design decision needs to be locked in — usually before either a plan or a journal.
- Cross-references between docs use markdown relative links (e.g., `[2026-05-25-cell5-road-pip-prior.md](journals/2026-05-25-cell5-road-pip-prior.md)`).
- Cited code/data paths use the post-2026-05-28 reorg layout (e.g., `runs/v3/training/loss_aug/...`, `runs/v3/tournaments/e10c_*/`).

## Where the heavy data lives

After the 2026-05-28 reorg:

- **`runs/v3/`** at `mcts_study/runs/v3/` is a symlink to `/home/chitii/catan_data/runs/v3/` on WSL Linux fs (off C: drive).
- WSL Python (training, tournaments, analyses) resolves this transparently.
- Windows-side tools see a broken symlink; use `\\wsl.localhost\Ubuntu\home\chitii\catan_data\runs\v3\` to browse from Windows.
- See [INDEX.md cross-reference tables](INDEX.md) for tournament dir locations.
