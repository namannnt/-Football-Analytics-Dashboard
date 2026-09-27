# Football Analytics Dashboard

An end-to-end football analytics project that separates raw delivery data, batch warehouse work, distributed curation, a fast serving database, modeling, and stakeholder exports.

## Architecture

```text
combined.csv / playerdata.csv / fresh weekly feeds
                    │
                    ▼
              ingest_dag.py (Airflow)
                    │
     1. ingest_raw_data ───────────────► HDFS + Hive external raw tables
                    │
     2. hive_batch_aggregate ──────────► season standings and player rolling stats
                    │
     3. spark_hive_read ───────────────► Spark SQL joins team and player grains
                    │
     4. load_to_postgres ──────────────► indexed PostgreSQL serving layer
                    │
     5. refresh_stored_procedures ─────► weekly standings and rolling pace
                    │
     6. run_anomaly_flags ─────────────► robust anomaly flags
                    │
                    ▼
      features.py → model.py → ab_testing.py
                    │
                    ▼
       charts.py → insights.py → exports
                    │
                    ▼
                  app.py
```

At this data volume, Hive/HDFS is more infrastructure than the dataset needs — this layer exists to demonstrate the batch-warehouse pattern MiQ's stack likely uses for raw delivery data (the JD names Qubole/Databricks/Spark explicitly), not because a few thousand rows require distributed storage. In production, this is the layer real daily delivery volume across many advertisers/teams would actually live in before being curated down.

## Pipeline layers

### Raw and Hive batch warehouse

[`ingest_raw_data.py`](ingest_raw_data.py) accepts `combined.csv` or the included `combined (1).csv`, plus `playerdata.csv`. It keeps immutable source snapshots under an HDFS run ID and refreshes stable canonical CSV locations used by Hive external tables.

[`hive_queries.hql`](hive_queries.hql) is the batch entry point:

- [`external_raw_tables.hql`](hive_queries/external_raw_tables.hql) defines the external raw tables.
- [`season_standings_raw.hql`](hive_queries/season_standings_raw.hql) calculates cumulative points, wins, losses, goals, and goal difference per real matchday.
- [`player_rolling_stats_raw.hql`](hive_queries/player_rolling_stats_raw.hql) calculates five-appearance per-90 player windows.

The included match CSV has results but no season, date, or matchday fields. Ingestion preserves this honestly as season `unknown` with a null matchday; it does not pretend file order is a matchday. Supply a current match feed with `season` and `matchday` columns to populate standings. The included `playerdata.csv` is a player profile snapshot rather than match-level event data, so rolling player output is empty until the player source also contains `match_id`, `matchday`, `team`, `minutes`, and the event metrics.

### Spark and PostgreSQL

[`spark_layer.py`](spark_layer.py) uses Hive-enabled Spark SQL to join the warehouse tables on season, team, and matchday. It writes partitioned Parquet first, then publishes the curated result to `analytics.team_player_matchday` through Spark JDBC.

[`stored_procedures.sql`](stored_procedures.sql) creates indexed serving tables for weekly standings and rolling points pace. [`anomaly_flagging.py`](anomaly_flagging.py) writes robust median absolute deviation flags after the refresh.

### Analytics and delivery

- [`features.py`](features.py) calculates shifted pre-match form, pre-match ELO, home advantage, and prior matchup record so future results cannot leak into a fixture's features.
- [`model.py`](model.py) trains an On-Pace versus At-Risk logistic classifier from first-half-of-season observations only.
- [`ab_testing.py`](ab_testing.py) performs Welch comparisons with effect size.
- [`charts.py`](charts.py) supplies radar, pace, distribution, and correlation visuals.
- [`insights.py`](insights.py) produces deterministic stakeholder takeaways.
- [`export.py`](export.py), [`excel_export.py`](excel_export.py), and [`pptx_export.py`](pptx_export.py) generate a PDF brief, formatted workbook, and PowerPoint deck.
- [`powerbi_dataset.py`](powerbi_dataset.py) emits the serving tables as CSV or Parquet for Power BI.
- [`app.py`](app.py) is the interactive Streamlit layer over PostgreSQL.

The original single-file prototype remains in [`final.py`](final.py) for comparison.

The Streamlit tabs expose the pacing trend, normalized standings radar, metric distributions, correlation heatmap, and Welch group comparison. The statistical view reports sample sizes, means, mean difference, t-statistic, p-value, and Cohen's d while keeping statistical and practical significance separate.

Power BI extracts are an explicit CLI operation rather than a scheduled pipeline side effect:

```bash
python powerbi_dataset.py --output powerbi --format parquet
```

### Model training and inference

[`ml_dataset.py`](ml_dataset.py) converts fixtures into one observation per team before each kickoff. Form, ELO, opponent ELO, prior matchup record, venue, and season progress use only results available before that fixture. A team is labelled `On-Pace` when it finishes in the top half of its season table, ordered by final points, goal difference, goals scored, and team name.

Training uses first-half observations and holds out the newest season for chronological evaluation:

```bash
python train_model.py --source demo
```

The demo evaluation is explicitly marked demonstration-only because it is synthetic and small. Model training is manual and does not run in the serving DAG. After PostgreSQL refresh, Airflow builds current pre-match features and runs inference only when the configured model and metadata artifacts exist; otherwise the scoring task exits successfully with a skipped status. Predictions are written to `analytics.team_pace_predictions` and displayed by Streamlit when available.

## Configuration

Copy `.env.example` into the environment used by Airflow and Spark. Secrets should come from your orchestrator or secret manager in a real deployment.

```bash
pip install -r requirements.txt
```

The infrastructure runtime needs Hadoop/HDFS, HiveServer2 with Beeline, Spark with Hive support, PostgreSQL, and Airflow. The Airflow worker must be able to invoke `hdfs`, `beeline`, and `spark-submit`.

Run the weekly DAG on its default Monday 04:00 schedule, or set `FOOTBALL_DAG_SCHEDULE` to a daily cron expression such as `0 4 * * *`.

After the serving tables exist:

```bash
streamlit run app.py
```

## Source contract

The normalizer accepts common aliases, but the preferred match schema is:

```text
season, matchday, match_id, kickoff_ts, home_team, away_team, home_goals, away_goals
```

The preferred match-level player schema is:

```text
season, matchday, match_id, player_id, player_name, team, minutes,
goals, assists, shots, tackles, passes_completed
```
