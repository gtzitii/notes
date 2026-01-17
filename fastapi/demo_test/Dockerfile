# 使用官方 Python 轻量镜像
FROM python:3.10-slim

# 设置容器内工作目录
WORKDIR /app

# 安装运行 FastAPI 所需的依赖
# --no-cache-dir 减少镜像体积
RUN pip install --no-cache-dir fastapi uvicorn

# 把当前目录下的代码复制到容器中
COPY main.py /app/main.py

# 容器启动时执行的命令
# 使用 uvicorn 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]