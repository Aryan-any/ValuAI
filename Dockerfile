# # 1️⃣ Start from a lightweight Linux with Python
# FROM python:3.10-slim

# # 2️⃣ Set working directory inside container
# WORKDIR /app

# # 3️⃣ Copy requirements first (for caching)
# COPY requirements.base.txt .
# COPY requirements.ml.txt .

# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# ENV PIP_NO_DEPS=1

# # 4️⃣ Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.base.txt \
# && pip install --no-cache-dir --no-deps -r requirements.ml.txt


# # 5️⃣ Copy your entire project into container
# COPY . .

# # 6️⃣ Expose port (FastAPI runs on 8000)
# EXPOSE 8000

# # 7️⃣ Command to start your API
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Load environment variables at runtime
ENV ENV=production


WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy app
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

