FROM eclipse-temurin:11-jre-jammy

# Match the Hadoop client bundled into the Airflow/Spark image.
ARG HADOOP_VERSION=3.4.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl procps \
    && curl -fsSL "https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz" \
       | tar -xz -C /opt \
    && ln -s "/opt/hadoop-${HADOOP_VERSION}" /opt/hadoop \
    && rm -rf /var/lib/apt/lists/*

ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=/opt/hadoop/bin:/opt/hadoop/sbin:${PATH}

COPY infra/hadoop/conf/ /opt/hadoop/etc/hadoop/
COPY infra/hadoop/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 9000 9864 9870
ENTRYPOINT ["/entrypoint.sh"]
