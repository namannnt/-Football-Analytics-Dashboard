-- Typed, columnar match history for Spark and PostgreSQL serving.
DROP TABLE IF EXISTS hive_db.match_history;

CREATE TABLE hive_db.match_history
STORED AS ORC AS
SELECT season,
       CAST(matchday AS INT) AS matchday,
       match_id,
       kickoff_ts,
       home_team,
       away_team,
       CAST(home_goals AS INT) AS home_goals,
       CAST(away_goals AS INT) AS away_goals
FROM hive_db.matches_raw
WHERE CAST(matchday AS INT) IS NOT NULL
  AND CAST(home_goals AS INT) IS NOT NULL
  AND CAST(away_goals AS INT) IS NOT NULL;
