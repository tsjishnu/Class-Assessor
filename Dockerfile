# Hugging Face Space (Docker SDK) / any container host. Listens on 7860, the port Spaces expects.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ASSESSOR_CACHE=/tmp/assessor-cache \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY --chown=user . .

# Bake the models into the image (a few hundred MB) so the first request does not download them.
RUN python scripts/prefetch_models.py

EXPOSE 7860
# XSRF/CORS are disabled because the Space page embeds the app in an iframe, which otherwise breaks file uploads (HTTP 403).
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--server.maxUploadSize=1024", "--browser.gatherUsageStats=false"]
