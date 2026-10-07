# OpenMed 智能问诊分流系统 — Docker 多阶段构建
# 目标：生产镜像尽量精简，开发镜像包含调试工具

# ── 阶段 1：基础环境 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# ── 阶段 2：安装 Python 依赖 ──────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
# torch 只用于 Cross-Encoder 精排的 CPU 推理，先装 CPU-only 版本，
# 避免 sentence-transformers 拉取默认的 CUDA/cuDNN 依赖（几个 GB，且用不到）。
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 不再预下载 ChromaDB 内置英文 ONNX 模型（all-MiniLM-L6-v2）：
# 生产链路 embedding 统一由 chromadb 服务端计算（collection 创建时配置中文
# 模型 bge-small-zh-v1.5），app 容器客户端不需要本地模型缓存；且该模型托管在
# chroma-onnx-models.s3.amazonaws.com，国内网络拉不动，会导致构建失败。

# ── 阶段 3：生产镜像 ──────────────────────────────────────────────────────────
FROM base AS production

# 非 root 用户运行。先创建用户，后续 COPY 直接带 owner，避免 chown -R 复制出额外大层。
RUN useradd -m -u 1000 openmed

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=openmed:openmed . .

# 创建必要目录，只调整运行期需要写入的目录权限，避免递归 chown 整个应用。
RUN mkdir -p /app/data/chroma /app/logs /app/config && \
    chown openmed:openmed /app/data /app/data/chroma /app/logs /app/config
USER openmed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5).read()"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── 阶段 4：开发镜像 ──────────────────────────────────────────────────────────
FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    chmod -R 777 /app/data /app/logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
