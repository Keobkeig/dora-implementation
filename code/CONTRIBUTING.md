# Contributing

This repo is set up for team-based CS4782 final project work.

## Branching and PR Workflow

- Branch from `main` using `feature/<name>` or `fix/<name>`.
- Keep PRs focused and small (one main idea per PR).
- Add a clear PR description with:
  - objective,
  - files changed,
  - how you validated (tests, scripts, screenshots).

## Environment Setup (uv)

```bash
uv python install 3.11
uv sync --frozen --extra dev --extra notebooks
```

Use `uv.lock` as source of truth. If dependencies change:

```bash
uv lock
uv sync --frozen --extra dev --extra notebooks
```

## Code Quality

Run before pushing:

```bash
make test
```

## Notebook Workflow

- Put notebooks in `notebooks/`.
- Keep names descriptive: `YYYY-MM-DD_<topic>.ipynb`.
- Clear notebook outputs before committing when possible for cleaner diffs.
- Export final figures to `experiments/<run>/figures/`.

## Project Areas and Ownership

- `dora/`, `training/`, `benchmarks/`: core method and infrastructure.
- `experiments/`: configs, run metadata, result tables/plots.
- `demo/`: interactive demo app and demo tests.
- `presentation/`: LaTeX beamer slides.

## Suggested PR Labels

- `infra`
- `experiments`
- `demo`
- `slides`
- `bugfix`

## Commit Style

Use imperative commit messages:

- `add glue runner for sst2/mrpc/qqp`
- `fix dora benchmark import path`
- `update beamer slides with rank sweep results`
