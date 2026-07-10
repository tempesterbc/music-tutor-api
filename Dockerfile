FROM python:3.11-slim
# libsndfile for soundfile, ffmpeg for mp3/m4a decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
