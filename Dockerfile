FROM traefik/whoami as builder

FROM python:3.11.0-slim-bullseye as base
COPY --from=builder /whoami /whoami
RUN apt-get update && apt-get install -y vim curl ffmpeg
RUN pip install insanely-fast-whisper

RUN pip install fastapi uvicorn[standard] jinja2
RUN mkdir -p /data/hf_cache
RUN mkdir -p /data/mp3_cache
RUN pip install yt-dlp

ADD server.py /server.py
ADD schema.sql /schema.sql
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
