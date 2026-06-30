FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Pre-download model during build so runtime has no internet dependency
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('TaylorAI/bge-micro-v2')"

ENTRYPOINT ["python", "main.py"]
