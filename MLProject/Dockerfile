FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    mlflow==2.17.2 \
    numpy==2.0.2 \
    cloudpickle==3.1.2 \
    psutil==7.2.2 \
    scipy==1.13.1

EXPOSE 5000

CMD ["python", "modelling.py"]