ARG SPARK_VERSION=4.0.0
ARG HADOOP_VERSION=3.4.1
FROM spark:${SPARK_VERSION}-python3

ARG SPARK_VERSION
ARG HADOOP_VERSION
LABEL org.opencontainers.image.description="Spark ${SPARK_VERSION} with Hadoop ${HADOOP_VERSION} compatibility contract"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -o /opt/spark/jars/postgresql-42.7.4.jar https://jdbc.postgresql.org/download/postgresql-42.7.4.jar \
    && rm -rf /var/lib/apt/lists/*
