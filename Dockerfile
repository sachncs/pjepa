# Multi-stage build for pjepa.
# Stage 1: dependencies (cached unless pyproject.toml changes)
FROM python:3.12-slim AS deps
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: test (adds dev dependencies and runs tests)
FROM deps AS test
RUN pip install --no-cache-dir pytest pytest-cov pytest-xdist
RUN pjepa doctor

# Stage 3: default (runs the CLI)
FROM deps AS app
WORKDIR /app
COPY experiments ./experiments
COPY tests ./tests
COPY Makefile mkdocs.yml .pre-commit-config.yaml ./
COPY docs ./docs
COPY configs ./configs
ENTRYPOINT ["pjepa"]
CMD ["--help"]