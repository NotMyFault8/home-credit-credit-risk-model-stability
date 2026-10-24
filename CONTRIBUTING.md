# Contributing

Create a virtual environment, install `pip install -e '.[dev]'`, and run:

```bash
ruff check src tests scripts
pytest -q
home-credit demo --output artifacts/demo
```

Keep each source's real grain explicit. Add tests when changing feature joins, temporal splits, scoring contracts, or model activation. Do not use the final holdout to choose parameters. Do not commit competition data, secrets, generated models, or per-application predictions. Label any synthetic results clearly and retain unfavorable measured results.
