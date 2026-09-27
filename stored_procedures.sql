CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.pipeline_refresh_log (
    refresh_id BIGSERIAL PRIMARY KEY,
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL
);

CREATE OR REPLACE PROCEDURE analytics.refresh_football_serving()
LANGUAGE plpgsql
AS $$
BEGIN
    IF to_regclass('analytics.team_player_matchday') IS NULL THEN
        RAISE EXCEPTION 'analytics.team_player_matchday does not exist; load Spark output first';
    END IF;

    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_team_player_matchday_lookup
             ON analytics.team_player_matchday (season, team, matchday)';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_team_player_matchday_player
             ON analytics.team_player_matchday (player_id, season, matchday)';

    DROP TABLE IF EXISTS analytics.weekly_standings;
    CREATE TABLE analytics.weekly_standings AS
    SELECT DISTINCT season, matchday, team, team_played AS played,
           team_wins AS wins, team_draws AS draws, team_losses AS losses,
           team_goals_for AS goals_for, team_goals_against AS goals_against,
           team_goal_difference AS goal_difference, team_points AS points,
           DENSE_RANK() OVER (
               PARTITION BY season, matchday
               ORDER BY team_points DESC, team_goal_difference DESC, team_goals_for DESC
           ) AS standing
    FROM analytics.team_player_matchday;

    DROP TABLE IF EXISTS analytics.rolling_pace;
    CREATE TABLE analytics.rolling_pace AS
    SELECT season, matchday, team, MAX(team_points) AS points,
           MAX(team_played) AS played,
           CASE WHEN MAX(team_played) > 0
                THEN MAX(team_points)::numeric / MAX(team_played) END AS points_per_match,
           CASE WHEN MAX(team_played) > 0
                THEN (MAX(team_points)::numeric / MAX(team_played)) * 38 END AS projected_38_game_points
    FROM analytics.team_player_matchday
    GROUP BY season, matchday, team;

    CREATE INDEX IF NOT EXISTS idx_weekly_standings_lookup
        ON analytics.weekly_standings (season, matchday, standing);
    CREATE INDEX IF NOT EXISTS idx_rolling_pace_lookup
        ON analytics.rolling_pace (season, team, matchday);
    INSERT INTO analytics.pipeline_refresh_log(status) VALUES ('success');
END;
$$;
