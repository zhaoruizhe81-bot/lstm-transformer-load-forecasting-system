# Repository Guidelines

## Project Structure & Module Organization

The application lives under `code/`. `app.py` is the Flask backend entry point and wires API routes to `services/`. Database models are in `models/database.py`, response and chart helpers are in `utils/`, and forecasting logic is in `algorithms/` (`lstm_model.py`, `transformer_model.py`, `hybrid_model.py`, `preprocessing.py`). The Streamlit frontend is in `code/frontend/`, with page implementations in `frontend/views/` and shared client helpers in `frontend/ui_utils/`. Data and model artifacts are stored in `code/data/`, `code/ETT-small/`, `code/saved_models/`, and generated analysis outputs in `code/images/`.

## Build, Test, and Development Commands

Run commands from `code/` unless noted.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mysql -u root -p < database.sql
python app.py
```

These create a local environment, install backend dependencies, initialize MySQL, and start Flask at `http://localhost:5000`.

```bash
cd frontend
pip install -r requirements.txt
streamlit run main.py
```

This starts the frontend at `http://localhost:8501`; keep the backend running first. Use `python test_api.py` as the current API smoke check. Use `python train_all.py`, `python evaluate_and_plot.py`, or `python update_predictions.py` only when you need to refresh models, metrics, or prediction outputs.

## Coding Style & Naming Conventions

Use Python 3.8+ with 4-space indentation. Keep module names lowercase with underscores, class names in `PascalCase`, functions and variables in `snake_case`, and constants in uppercase. Follow the existing service-oriented layout: route handlers should stay thin, while business logic belongs in `services/` and model code belongs in `algorithms/`. Prefer explicit imports and short Chinese comments only where they clarify non-obvious behavior.

## Testing Guidelines

There is no formal pytest suite yet. Add focused tests as `test_*.py` near the related module or under a future `tests/` directory. For API changes, include a smoke path covering login, permission-sensitive routes, and database reads/writes. Verify manually with curl or Streamlit when changing request/response shapes.

## Commit & Pull Request Guidelines

This folder has no local Git history, so use clear imperative commits with optional scopes, for example `backend: validate prediction task dates` or `frontend: handle expired sessions`. Pull requests should describe the change, list test commands and results, note database/schema or model artifact changes, and include screenshots for Streamlit UI updates.

## Security & Configuration Tips

Database credentials are currently hardcoded for local MySQL (`root:123456`). Do not commit real credentials, private datasets, large regenerated model files, or environment-specific paths without confirming they are required.
