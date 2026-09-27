#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /data/name/current/VERSION ]; then
  hdfs namenode -format -force -nonInteractive
fi

hdfs --daemon start namenode
hdfs --daemon start datanode

for attempt in $(seq 1 30); do
  if hdfs dfsadmin -safemode wait >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

hdfs dfs -mkdir -p /football/raw /football/curated /user/hive/warehouse
hdfs dfs -chmod -R 777 /football /user/hive/warehouse
tail -F /opt/hadoop/logs/*namenode*.log /opt/hadoop/logs/*datanode*.log
