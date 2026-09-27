-- Canonical CSVs are produced during ingestion; original source files are retained separately.
CREATE DATABASE IF NOT EXISTS hive_db;
DROP TABLE IF EXISTS hive_db.matches_raw;
CREATE EXTERNAL TABLE hive_db.matches_raw (
    season STRING, matchday STRING, match_id STRING, kickoff_ts STRING,
    home_team STRING, away_team STRING, home_goals STRING, away_goals STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"', 'escapeChar'='\\')
STORED AS TEXTFILE
LOCATION '${hivevar:football_matches_path}';

DROP TABLE IF EXISTS hive_db.players_raw;
CREATE EXTERNAL TABLE hive_db.players_raw (
    player_id STRING, full_name STRING, positions STRING,
    nationality STRING, overall_rating STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"', 'escapeChar'='\\')
STORED AS TEXTFILE
LOCATION '${hivevar:football_players_path}';

DROP TABLE IF EXISTS hive_db.player_match_stats_raw;
CREATE EXTERNAL TABLE hive_db.player_match_stats_raw (
    season STRING, matchday STRING, match_id STRING, player_id STRING,
    player_name STRING, team STRING, minutes STRING, goals STRING,
    assists STRING, shots STRING, tackles STRING, passes_completed STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ('separatorChar'=',', 'quoteChar'='"', 'escapeChar'='\\')
STORED AS TEXTFILE
LOCATION '${hivevar:football_player_stats_path}';
