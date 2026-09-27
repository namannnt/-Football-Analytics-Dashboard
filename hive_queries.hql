-- Full Hive batch entry point. Run through Hive CLI/Beeline with the hivevar paths
-- documented in README.md. Individual query files are kept in hive_queries/ for review.
SOURCE hive_queries/external_raw_tables.hql;
SOURCE hive_queries/match_history_raw.hql;
SOURCE hive_queries/season_standings_raw.hql;
SOURCE hive_queries/player_rolling_stats_raw.hql;
