-- Per-player rates over the latest five appearances, using match-level event data.
DROP TABLE IF EXISTS hive_db.player_rolling_stats;

CREATE TABLE hive_db.player_rolling_stats
STORED AS ORC AS
WITH appearance_rows AS (
    SELECT season, CAST(matchday AS INT) AS matchday, match_id, player_id,
           player_name, team, CAST(minutes AS DOUBLE) AS minutes,
           CAST(COALESCE(NULLIF(TRIM(goals), ''), '0') AS DOUBLE) AS goals,
           CAST(COALESCE(NULLIF(TRIM(assists), ''), '0') AS DOUBLE) AS assists,
           CAST(COALESCE(NULLIF(TRIM(shots), ''), '0') AS DOUBLE) AS shots,
           CAST(COALESCE(NULLIF(TRIM(tackles), ''), '0') AS DOUBLE) AS tackles,
           CAST(COALESCE(NULLIF(TRIM(passes_completed), ''), '0') AS DOUBLE) AS passes_completed
    FROM hive_db.player_match_stats_raw
    WHERE CAST(minutes AS DOUBLE) > 0 AND CAST(matchday AS INT) IS NOT NULL
), rolling AS (
    SELECT *,
           SUM(minutes) OVER recent_five AS minutes_last_5,
           SUM(goals) OVER recent_five AS goals_last_5,
           SUM(assists) OVER recent_five AS assists_last_5,
           SUM(shots) OVER recent_five AS shots_last_5,
           SUM(tackles) OVER recent_five AS tackles_last_5,
           SUM(passes_completed) OVER recent_five AS passes_completed_last_5
    FROM appearance_rows
    WINDOW recent_five AS (
        PARTITION BY season, player_id ORDER BY matchday, match_id
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    )
)
SELECT season, matchday, match_id, player_id, player_name, team,
       minutes_last_5,
       90.0 * goals_last_5 / minutes_last_5 AS goals_per90_last5,
       90.0 * assists_last_5 / minutes_last_5 AS assists_per90_last5,
       90.0 * shots_last_5 / minutes_last_5 AS shots_per90_last5,
       90.0 * tackles_last_5 / minutes_last_5 AS tackles_per90_last5,
       90.0 * passes_completed_last_5 / minutes_last_5 AS passes_completed_per90_last5
FROM rolling;
