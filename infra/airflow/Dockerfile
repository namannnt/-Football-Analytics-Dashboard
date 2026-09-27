FROM apache/airflow:2.10.5-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG HADOOP_VERSION=3.4.1
ARG HIVE_VERSION=4.0.1
ARG SPARK_VERSION=4.0.0

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl openjdk-17-jre-headless procps \
    && curl -fsSL "https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz" | tar -xz -C /opt \
    && curl -fsSL "https://archive.apache.org/dist/hive/hive-${HIVE_VERSION}/apache-hive-${HIVE_VERSION}-bin.tar.gz" | tar -xz -C /opt \
    && curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" | tar -xz -C /opt \
    && ln -s "/opt/hadoop-${HADOOP_VERSION}" /opt/hadoop \
    && ln -s "/opt/apache-hive-${HIVE_VERSION}-bin" /opt/hive \
    && ln -s "/opt/spark-${SPARK_VERSION}-bin-hadoop3" /opt/spark \
    && curl -fsSL -o /opt/spark/jars/postgresql-42.7.4.jar https://jdbc.postgresql.org/download/postgresql-42.7.4.jar \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV HIVE_HOME=/opt/hive
ENV HIVE_CONF_DIR=/opt/hive/conf
ENV SPARK_HOME=/opt/spark
ENV PATH=/opt/hadoop/bin:/opt/hive/bin:/opt/spark/bin:${PATH}

COPY infra/hadoop/conf/ /opt/hadoop/etc/hadoop/
COPY infra/hive/conf/ /opt/hive/conf/
COPY infra/spark/conf/ /opt/spark/conf/
COPY requirements-runtime.txt /tmp/requirements-runtime.txt

USER airflow
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-runtime.txt
