FROM spark:4.0.0-python3

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -o /opt/spark/jars/postgresql-42.7.4.jar https://jdbc.postgresql.org/download/postgresql-42.7.4.jar \
    && rm -rf /var/lib/apt/lists/*
