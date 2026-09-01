# Policy Portfolio QBR Dashboard

A local, CXO-level Streamlit dashboard for multi-million-row insurance policy
data. It separates the one-time data preparation workload from interactive
analysis:

```text
Raw CSV / Parquet → prepare_data.py → compact DuckDB mart → Streamlit dashboard
                                             ↓
                                  aggregate-only context → Ollama
```

## What it delivers

- Month-end active population based on distinct `beneficiarykey` and policy
  membership dates.
- Mobile App penetration using distinct `registereduserkey` ÷ distinct
  `beneficiarykey`, plus linked-beneficiary coverage to identify household or
  one-to-many account effects.
- GP, NP, and TPA fee evaluation by underwriting year, payer, and policy type.
- A guarded Gen BI page: a deterministic semantic router maps each question to
  only the relevant aggregate metrics, then a local Ollama model produces a
  concise MBB-style CXO narrative.

## Quick start (Windows / PowerShell)

Create a project virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Generate a safe, all-dimension demo extract when the production file is not
available. It creates 288 policy-year rows spanning six countries/payers,
providers, contracts, policy and network types, demographics, membership
timing, and mobile-registration patterns:

```powershell
python generate_demo_data.py
```

Stage the supplied full extract as `data/policies.csv` or `data/policies.parquet`,
then build the reusable analytics mart:

```powershell
python prepare_data.py --source .\data\policies_demo.csv
```

Launch the dashboard:

```powershell
python -m streamlit run app.py
```

## Ollama setup

Install Ollama using your normal operating-system setup and pull the model
configured in `.env`. The dashboard never starts, stops, or otherwise manages
the Ollama service; the configured endpoint must already be available. It does
not pull models automatically.

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:1b
```

For example:

```powershell
ollama pull llama3.2:1b
python -m streamlit run app.py
```

Use **Check model response** in the Gen BI page to send the configured model a
small `READY` prompt. The same probe runs only after a Gen BI narration fails,
so it distinguishes an unavailable model from a prompt-specific problem without
adding model work to normal dashboard reruns.

For a Lenovo P5 with 128 GB RAM and no GPU, use a small quantized 1–3B model,
keep it warm, limit the response to ~180 tokens, and retain the cache. The
dashboard's filter and metric calculations target sub-second warm latency;
CPU text generation itself is inherently variable and will usually take longer
than one second for a new answer. The app therefore prepares the
question-specific deterministic evidence pack before invoking the model, limits
the model briefing and completion size, and caches duplicate question-and-scope
requests for 15 minutes. If Ollama does not respond, Gen BI reports the probe
result and returns an evidence-bound deterministic fallback.

## Gen BI evaluation records

Each submitted Gen BI question writes one aggregate-only Parquet interaction
row under `data/gen_bi_evaluations/date=YYYY-MM-DD/`. The record includes the
UTC timestamp, question, semantic focus, selected sources and entities,
applied filter scope, question-specific evidence supplied to Ollama, all
available aggregate metric values, model answer/status, and timing measures.
The output is ignored by Git because it may contain internal business
questions and summaries. Set `GEN_BI_EVALUATION_DIR` to redirect the local
Parquet dataset.

## Data and privacy controls

- Raw extracts and the generated `*.duckdb` mart are ignored by Git.
- The Streamlit process receives only aggregated tables for visualizations.
- Ollama receives a compact aggregate briefing only—never beneficiary keys,
  policy-level records, or arbitrary model-written SQL.
- Gen BI evaluation records persist only aggregate metrics and local business
  question/answer text; they never include policy or beneficiary records.
- The app expects currency values to already be USD, as named in the source.

## Performance design

`prepare_data.py` is deliberately batch-oriented. It normalizes data types and
materializes a month-end active-membership table once, instead of expanding
three million policy rows on every click. At run time, DuckDB scans a slim
temporary scoped fact table once for the premium, payer, and adoption views.
The unfiltered active-population chart uses a pre-aggregated fast path;
filtered population is computed from the month-end membership table for
correct distinct-member semantics.

The exact latency depends on storage (local NVMe strongly recommended), data
cardinality, selected scope, and whether the mart is warm in RAM. The `Data
guide` tab reports live DuckDB query time so the target can be measured against
the actual 3M-row extract rather than assumed.
