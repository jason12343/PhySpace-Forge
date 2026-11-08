FROM python:3.11-slim

WORKDIR /app

# Install lightweight CPU PyTorch + NumPy + PhySpace-Forge package for MCP stdio introspection
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch numpy

COPY pyproject.toml README.md README_EN.md LICENSE ./
COPY physpace_forge ./physpace_forge
COPY assets ./assets

RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["physpace", "mcp"]
