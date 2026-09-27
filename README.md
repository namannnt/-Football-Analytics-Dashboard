# Football Analytics Dashboard

A portfolio-scale analytics system that shows how raw football delivery data can move through HDFS, Hive, Spark, PostgreSQL, leak-free model inference, an interactive Streamlit dashboard, and stakeholder exports.

The repository contains two kinds of input:

- `combined (1).csv` and `playerdata.csv` are the original legacy files. They remain unchanged and do not contain enough time-grained information to exercise the full pipeline.
- `data/demo_matches.csv` and `data/demo_player_match_stats.csv` are deterministic, synthetic validation data. They contain three seasons, four teams, ten matchdays per season, and repeated player appearances for five-match rolling windows. They are not historical results and are not evidence of model performance.

## Runtime architecture

```text
Production-style CSVs or synthetic demo CSVs
                    │
                    ▼
             ingest_raw_data.py
         validation + canonicalization
                    │
                    ▼
       HDFS archive and current raw zone
                    │
                    ▼
           Hive external raw tables
                    │
                    ▼
  Hive match history + standings + player windows
                    │
                    ▼
          Spark SQL team/player curation
                    │
                    ▼
          Partitioned Parquet curated layer
                    │
                    ▼
 analytics.matches + analytics.team_player_matchday
                   PostgreSQL
                    │
                    ▼
 stored procedures → anomaly flags → pre-match features
                    │
                    ▼
       saved model artifact → inference only
                    │
                    ▼
        analytics.team_pace_predictions
                    │
                    ▼
 Streamlit + statistical analysis + deterministic insights
                    │
                    ▼
       PDF + Excel + PowerPoint + Power BI files
```

The Airflow DAG runs these tasks in order:

```text
ingest_raw_data
  → hive_batch_aggregate
  → spark_hive_read
  → load_to_postgres
  → refresh_stored_procedures
  → run_anomaly_flags
  → build_ml_features
  → score_pace_model
```

Model training is a separate manual flow and is not part of every DAG run.

At this data volume, Hive/HDFS is more infrastructure than the dataset needs — this layer exists to demonstrate the batch-warehouse pattern MiQ's stack likely uses for raw delivery data (the JD names Qubole/Databricks/Spark explicitly), not because a few thousand rows require distributed storage. In production, this is the layer real daily delivery volume across many advertisers/teams would actually live in before being curated down.

## Why each storage and compute layer exists

- **HDFS** preserves immutable delivery snapshots and exposes stable current paths for batch processing.
- **Hive** applies schema to the raw files and calculates whole-history standings, typed match history, and five-appearance player windows in batch.
- **Spark** reads the Hive-managed aggregates, performs the cross-grain team/player join, writes partitioned Parquet, and publishes curated tables through JDBC.
- **PostgreSQL** serves indexed, low-latency tables for the dashboard, model scoring, statistical analysis, and exports.
- **Airflow** controls ordering, retries, and the weekly or configurable schedule.

This separation demonstrates a production data-delivery pattern. A single PostgreSQL database or local dataframe pipeline would be enough for the bundled data volume.

## Data contracts

Set `FOOTBALL_DATASET` to select input behavior:

| Value | Match source | Player event source | Validation |
| --- | --- | --- | --- |
| `demo` | Included synthetic CSV | Included synthetic CSV | Strict production-style contract |
| `custom` | `FOOTBALL_MATCH_CSV` | `FOOTBALL_PLAYER_EVENT_CSV` | Strict production-style contract |
| `legacy` | Original bundled files or configured overrides | Optional | Compatible aliases; absent time fields remain absent |

Canonical match columns:

```text
season, matchday, match_id, kickoff_ts,
home_team, away_team, home_goals, away_goals
```

Canonical match-level player columns:

```text
season, matchday, match_id, player_id, player_name, team,
minutes, goals, assists, shots, tackles, passes_completed
```

Strict inputs fail with the source row and missing or invalid fields. Player events must reference an existing season/match and one of its two teams. The legacy match file is assigned season `unknown` only when no season exists; row order is never treated as matchday. The legacy player file is a profile snapshot, so it cannot populate rolling per-90 output by itself.

Run ingestion locally without Hadoop to inspect the canonical files:

```bash
python scripts/run_demo_ingestion.py
```

Output is written under ignored `build/raw/` paths using the same schemas consumed by Hive.

## Hive and Spark flow

The DAG executes the Hive scripts individually:

- `hive_queries/external_raw_tables.hql` defines CSV-backed external tables.
- `hive_queries/match_history_raw.hql` creates typed ORC match history.
- `hive_queries/season_standings_raw.hql` calculates cumulative team standings by real matchday.
- `hive_queries/player_rolling_stats_raw.hql` calculates five-appearance per-90 windows.

`spark_layer.py` reads `hive_db.season_standings`, `hive_db.player_rolling_stats`, and `hive_db.match_history` through Hive-enabled Spark SQL. It writes two Parquet datasets and then publishes `analytics.team_player_matchday` and `analytics.matches` to PostgreSQL.

`stored_procedures.sql` produces `analytics.weekly_standings` and `analytics.rolling_pace`, with serving indexes. `anomaly_flagging.py` adds robust median absolute deviation flags without inventing alerts when the group spread is zero.

## Model methodology

### Training flow

```text
Completed historical seasons
  → one team row before every kickoff
  → first-half observations only
  → oldest seasons train / newest season evaluates
  → median imputation + scaling + logistic regression
  → versioned joblib artifact + JSON metadata
```

Features are calculated before kickoff:

- mean points and goal difference over prior matches;
- team and opponent pre-match ELO;
- ELO difference;
- prior head-to-head points;
- home indicator;
- season progress based on the known schedule length.

The target is `On-Pace` when a team finishes in the top half of its final season table. Final ordering uses points, goal difference, goals scored, then team name for a deterministic tie break. Final-season values are labels only and are never model inputs.

Training and evaluation use different seasons. No random row split is used. Train manually:

```bash
python train_model.py --source demo
```

The command reports accuracy, precision, recall, F1, ROC-AUC when defined, and a confusion matrix. It marks the demo metrics as demonstration-only because the dataset is synthetic and the holdout is small.

### Inference flow

```text
analytics.matches
  → current pre-match features
  → latest eligible first-half row per team
  → saved artifact and matching feature metadata
  → analytics.team_pace_predictions
```

Set `FOOTBALL_SEASON_MATCHDAYS` to the expected competition length. This prevents an incomplete live season from being mistaken for a completed schedule. When no trained artifact exists, Airflow records a successful skipped result and the Streamlit application shows a clear no-predictions message.

## Dashboard and exports

`app.py` reads PostgreSQL and provides:

- latest standings and model predictions;
- pacing trends;
- normalized team radar comparison;
- metric distributions and a correlation heatmap;
- Welch group comparisons with sample sizes, means, mean difference, t-statistic, p-value, and Cohen's d;
- deterministic standings, trend, anomaly, and model summaries;
- PDF, Excel, and PowerPoint downloads.

The statistical view treats its results as descriptive evidence. Statistical significance and practical significance are shown as separate considerations.

Power BI output is an explicit CLI operation:

```bash
python powerbi_dataset.py --output powerbi --format parquet
```

## Local Docker demonstration

Requirements: Docker Engine with Compose, at least 8 GB of available memory, and enough disk space for Hadoop, Hive, Spark, and Airflow images.

```bash
cp .env.example .env
# Edit the local demonstration password and any ports or schedules.
docker compose up -d --build
docker compose ps
```

The stack contains a compact HDFS node, standalone Hive metastore, HiveServer2, one Spark master and worker, PostgreSQL, single-node Airflow, and Streamlit. It demonstrates integration boundaries and is not a resilient cluster.

Train the demonstration model and trigger the DAG:

```bash
docker compose exec airflow python /opt/airflow/dags/repo/train_model.py --source demo \
  --model /opt/airflow/dags/repo/artifacts/on_pace.joblib \
  --metadata /opt/airflow/dags/repo/artifacts/on_pace.metadata.json
docker compose exec airflow airflow dags trigger football_raw_to_serving
docker compose exec airflow airflow dags list-runs -d football_raw_to_serving
```

Then open:

| Service | Address |
| --- | --- |
| Streamlit | `http://localhost:8501` |
| Airflow | `http://localhost:8088` |
| Spark master | `http://localhost:8080` |
| HDFS NameNode | `http://localhost:9870` |
| HiveServer2 | `localhost:10000` |
| PostgreSQL | `localhost:5432` |

The first build downloads the pinned client distributions. The local PostgreSQL password in `.env.example` is a demonstration default and must be replaced outside local Compose.

## Native Python setup and tests

Use Python 3.11 for the complete Airflow dependency set. Lightweight analytics and unit tests can use:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-test.txt
python -m pytest -m "not integration"
```

Run compilation and the full suite, including the expected integration skip when no stack is active:

```bash
python -m compileall -q .
pytest
```

After the Docker DAG has succeeded, opt into the database smoke test from the host:

```bash
RUN_DISTRIBUTED_E2E=1 python -m pytest -m integration
```

GitHub Actions runs the unit subset on Python 3.11. It does not start Hadoop, Hive, Spark, or PostgreSQL.

## Known limitations

- The synthetic dataset validates contracts and orchestration mechanics; it cannot support meaningful football conclusions or model performance claims.
- The legacy files cannot populate real matchday standings and player rolling statistics together.
- The Docker topology has one node per system, weak local authentication, and no high availability.
- This development host does not have Docker, so the Compose services and distributed DAG have not been executed here. Static topology checks and unit tests do not constitute a distributed end-to-end pass.
- Spark JDBC overwrites serving tables in two operations. A production implementation should stage tables and swap them transactionally.
- The model uses a small linear baseline. Production work would add more seasons, calibration, drift checks, competition-aware schedule length, and a registry-driven promotion process.
- Statistical comparisons use repeated matchday observations and should not be interpreted as causal experiments.

## Production improvements

- Store raw data in durable object storage with immutable manifests, checksums, and schema versioning.
- Replace local credentials with a secret manager and enable authenticated, encrypted service connections.
- Use a managed metastore and scalable Spark runtime with lineage and data-quality gates.
- Publish PostgreSQL tables atomically and retain run-level audit metadata.
- Add late-data handling, idempotent partitions, backfills, and alerting on row-count or freshness failures.
- Train from multiple real seasons, evaluate by competition and time, monitor drift, and require explicit model promotion.

## Historical prototype

`final.py` is intentionally retained as the original single-file Streamlit prototype. The operational application is `app.py`; the prototype's random split and upload-based flow are not part of the current pipeline.
