FROM python:3.12.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_LAUNCH_BROWSER=0

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
COPY models/mnist_cnn_albu.keras /app/mnist_cnn_albu.keras

EXPOSE 7860

CMD ["python", "app.py"]
