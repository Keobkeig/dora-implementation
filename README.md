# CS4782 Final Project: DoRA Reproduction

Team repository for the CS4782 final project on **DoRA (Weight-Decomposed Low-Rank Adaptation)**.

This repo is organized for collaborative work across:
- core DoRA implementation and experiments,
- presentation (LaTex beamer slides),
- demo app and demo-specific tests,
- notebook-heavy analysis with cleaner diffs.

## Repository Layout

```
dora_implementation/
├── dora/                    # Core DoRA package
├── scripts/                 # Training/benchmark entry scripts
├── training/                # Training utilities
├── benchmarks/              # Benchmark code
├── tests/                   # Unit/integration/benchmark tests
├── experiments/             # Experiment artifacts and configs
├── notebooks/               # Shared notebooks and analysis
├── demo/                    # Demo app + demo tests
├── presentation/            # LaTeX beamer slides
├── pyproject.toml           # Canonical dependency + tooling config
├── uv.lock                  # Fully locked environment
├── CONTRIBUTING.md          # Team workflow and contribution guide
└── README.md
```

## Quick Start (uv)

1) Install Python and dependencies (strict, lockfile-based):

```bash
uv python install 3.11
uv sync --frozen --extra dev --extra notebooks
```

2) Run tests:

```bash
uv run pytest -q
```

3) Start notebooks:

```bash
uv run jupyter lab
```

4) Build slides:

```bash
cd presentation
make
```

## Common Commands

```bash
# Lint + format
uv run ruff check .
uv run black .

# Run demo app
uv run python demo/gradio_app.py
```

## Make Targets

```bash
make format   # black + ruff --fix
make test     # format, then pytest
make clean    # remove caches/coverage artifacts
```

## Collaboration Workflow

- Create feature branches from `main`.
- Use small PRs with clear scope.
- Run lint/tests before pushing (`ruff`, `black`, `pytest`).
- Keep notebooks in `notebooks/` and avoid committing huge outputs.
- Keep experiment outputs in `experiments/` with dated run folders.

See `CONTRIBUTING.md` for detailed standards.

## Project Scope

Primary target for final report/presentation:
- LoRA vs DoRA on GLUE tasks (`sst2`, `mrpc`, `qqp`) with RoBERTa-base.
- Rank sweep and parameter-efficiency analysis.
- Demo comparing prediction behavior and efficiency metrics.

## Notes

- The lockfile is treated as source of truth for reproducibility.
- Python version is pinned via `.python-version` and `pyproject.toml`.
