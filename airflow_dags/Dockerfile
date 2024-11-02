FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && apt-get clean

USER airflow

COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY . /opt/airflow/

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["bash", "-c", "airflow webserver & airflow scheduler"]