-- Batch team standings by season and real matchday.
-- Rows without a source matchday are intentionally excluded: file order is not a matchday.
DROP TABLE IF EXISTS hive_db.season_standings;

CREATE TABLE hive_db.season_standings
STORED AS ORC AS
WITH team_match_rows AS (
    SELECT season, CAST(matchday AS INT) AS matchday, home_team AS team,
           1 AS played,
           CASE WHEN CAST(home_goals AS INT) > CAST(away_goals AS INT) THEN 1 ELSE 0 END AS wins,
           CASE WHEN CAST(home_goals AS INT) = CAST(away_goals AS INT) THEN 1 ELSE 0 END AS draws,
           CASE WHEN CAST(home_goals AS INT) < CAST(away_goals AS INT) THEN 1 ELSE 0 END AS losses,
           CAST(home_goals AS INT) AS goals_for, CAST(away_goals AS INT) AS goals_against,
           CASE WHEN CAST(home_goals AS INT) > CAST(away_goals AS INT) THEN 3
                WHEN CAST(home_goals AS INT) = CAST(away_goals AS INT) THEN 1 ELSE 0 END AS points
    FROM hive_db.matches_raw
    WHERE TRIM(matchday) <> '' AND CAST(matchday AS INT) IS NOT NULL
      AND TRIM(home_team) <> '' AND TRIM(away_team) <> ''
      AND CAST(home_goals AS INT) IS NOT NULL AND CAST(away_goals AS INT) IS NOT NULL
    UNION ALL
    SELECT season, CAST(matchday AS INT) AS matchday, away_team AS team,
           1 AS played,
           CASE WHEN CAST(away_goals AS INT) > CAST(home_goals AS INT) THEN 1 ELSE 0 END AS wins,
           CASE WHEN CAST(away_goals AS INT) = CAST(home_goals AS INT) THEN 1 ELSE 0 END AS draws,
           CASE WHEN CAST(away_goals AS INT) < CAST(home_goals AS INT) THEN 1 ELSE 0 END AS losses,
           CAST(away_goals AS INT) AS goals_for, CAST(home_goals AS INT) AS goals_against,
           CASE WHEN CAST(away_goals AS INT) > CAST(home_goals AS INT) THEN 3
                WHEN CAST(away_goals AS INT) = CAST(home_goals AS INT) THEN 1 ELSE 0 END AS points
    FROM hive_db.matches_raw
    WHERE TRIM(matchday) <> '' AND CAST(matchday AS INT) IS NOT NULL
      AND TRIM(home_team) <> '' AND TRIM(away_team) <> ''
      AND CAST(home_goals AS INT) IS NOT NULL AND CAST(away_goals AS INT) IS NOT NULL
), matchday_totals AS (
    SELECT season, matchday, team, SUM(played) AS played, SUM(wins) AS wins,
           SUM(draws) AS draws, SUM(losses) AS losses,
           SUM(goals_for) AS goals_for, SUM(goals_against) AS goals_against,
           SUM(points) AS points
    FROM team_match_rows
    GROUP BY season, matchday, team
)
SELECT season, matchday, team,
       SUM(played) OVER season_team AS played,
       SUM(wins) OVER season_team AS wins,
       SUM(draws) OVER season_team AS draws,
       SUM(losses) OVER season_team AS losses,
       SUM(goals_for) OVER season_team AS goals_for,
       SUM(goals_against) OVER season_team AS goals_against,
       SUM(goals_for - goals_against) OVER season_team AS goal_difference,
       SUM(points) OVER season_team AS points
FROM matchday_totals
WINDOW season_team AS (
    PARTITION BY season, team ORDER BY matchday
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
);
