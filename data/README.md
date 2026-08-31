# Data staging area

Place the supplied policy extract here before building the mart. CSV/TSV and
Parquet are supported. For example:

```powershell
python prepare_data.py --source .\data\policies.csv
```

The builder creates `data/policy_mart.duckdb`, which is intentionally ignored
by Git because it can contain sensitive policy and member data.

For a safe, rich synthetic file that exercises all portfolio and demographic
filters, run `python generate_demo_data.py`. It writes
`data/policies_demo.csv` and can be used with the same builder command.

Expected source columns use the names supplied in the sample data (for example
`beneficiarykey`, `registereduserkey`, `member_start_date`,
`gross_premium_usd`, `net_premium_usd`, and `tpa_usd`). The loader accepts
common underscore/case variants as well.
