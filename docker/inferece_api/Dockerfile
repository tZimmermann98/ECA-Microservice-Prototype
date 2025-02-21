FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    build-essential git ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /app

RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118

COPY hallo2/requirements.txt app/requirements.txt

RUN python3.10 -m pip install -r requirements.txt

RUN python3.10 -m pip install huggingface_hub

COPY inference_api/requirements.txt app/requirements.txt

RUN python3.10 -m pip install -r requirements.txt

COPY hallo2/hallo app/hallo
COPY hallo2/scripts/inferece_long.py app/scripts/inferece_long.py

COPY inference_api/defaut.yaml app/default.yaml
COPY inference_api/main.py app/main.py

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
