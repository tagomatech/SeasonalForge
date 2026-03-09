# Contributing

Thanks for contributing.

## Development setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
python -m unittest discover -s tests -v
```

## Pull request expectations

- Keep changes focused and scoped.
- Add or update tests for behavior changes.
- Keep strategy/data logic in `engine.py`/`data_provider.py` instead of UI code when possible.
- Document user-visible behavior changes in `README.md`.

## Style

- Prefer clear names over short names.
- Avoid hidden side effects in helper functions.
- Keep UI orchestration in `app.py` and reusable rendering logic in `ui_components.py`.
