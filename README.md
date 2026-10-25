# incident-ml-service

Ranks the lines of a failed Travis CI build log by how likely each is part of the failure explanation.
Trained on LogChunks; scope is Travis CI logs only.

## Setup
1. Put `line_model.json` and `metadata.json` from Colab's `model_out/` into `model/`.
2. In `requirements.txt`, pin xgboost to the `xgboost_version` in `metadata.json`.
3. `pip install -r requirements-dev.txt`
4. Run: `MODEL_API_KEY=dev-key uvicorn app.main:app --port 8000`
5. Test: `pytest -q tests`

## Contract v1.0
`GET /health` -> `{status, schema_version, model_version, n_features}`

`POST /predict` with header `X-API-Key`, body `{"schema_version":"1.0","log_text":"...","top_k":10}`
-> `{schema_version, model_version, n_lines, n_content_lines, travis_fold_markers, top_lines:[{line_no, score, section, text}], warnings:[]}`

Scores are comparable only inside one log. Use the ranking, not a global threshold.
Errors: 401 bad key, 413 too large, 422 invalid body, 503 busy (retry after `Retry-After`) or not configured.

## Environment
| Variable | Meaning | Default |
|---|---|---|
| MODEL_API_KEY | required secret | none (503 if unset) |
| MODEL_DIR | model folder | model |
| MAX_BODY_BYTES | request size cap | 8000000 |
| MAX_LINES | line cap | 200000 |
| MAX_CONCURRENT | simultaneous predictions | 2 |
| DISABLE_DOCS | `1` hides /docs | off |
| ALLOW_NO_AUTH | `1` skips the key (local only) | off |

## Deploy on Render
Push to GitHub (include `model/`), then New > Blueprint, set `MODEL_API_KEY`.
Free tier sleeps after 15 min idle; the first request after that takes about a minute.
