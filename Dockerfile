FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install vLLM for inference backend
RUN pip3 install --no-cache-dir vllm>=0.6.0

# Install ThinkBudget
COPY pyproject.toml README.md ./
COPY src/ src/
COPY dashboard/ dashboard/
COPY benchmarks/ benchmarks/

RUN pip3 install --no-cache-dir ".[all]"

# Generate benchmark datasets
RUN cd benchmarks/datasets && python3 generate_datasets.py

EXPOSE 8000 9100

# Default: start vLLM + ThinkBudget proxy
COPY cloudrift/run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
