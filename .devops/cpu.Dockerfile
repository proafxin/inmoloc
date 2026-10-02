ARG UBUNTU_VERSION=24.04
ARG NODE_VERSION=24

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

### Web UI, built from tools/ui
FROM docker.io/node:${NODE_VERSION} AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

### Build
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS build

ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential cmake git libssl-dev

ENV CC=gcc-14 CXX=g++-14

# OFF for a release: the version then has no -dev suffix
ARG BUILD_IS_DEV=ON

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# the CPU backend is built for every variant of the architecture and the one the host supports is loaded at run time
RUN if [ "$TARGETARCH" != "amd64" ] && [ "$TARGETARCH" != "arm64" ]; then \
        echo "Unsupported architecture: $TARGETARCH"; exit 1; \
    fi && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_IS_DEV=${BUILD_IS_DEV} && \
    cmake --build build --config Release -j$(nproc) --target local-inference-server local-inference-cli local-inference-app

RUN mkdir -p /app/lib /app/bin && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    cp build/bin/local-inference-server build/bin/local-inference-cli build/bin/local-inference /app/bin

### Run time
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS base

ARG BUILD_DATE
ARG APP_VERSION
ARG APP_REVISION
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="inmoloc" \
      org.opencontainers.image.description="Concurrent LLM serving on your own GPU (CPU build)" \
      org.opencontainers.image.url=https://github.com/proafxin/inmoloc \
      org.opencontainers.image.source=https://github.com/proafxin/inmoloc \
      org.opencontainers.image.licenses=MIT

# ffmpeg decodes video input for vision models
RUN apt-get update \
    && apt-get install -y libgomp1 libssl3 curl ffmpeg \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

WORKDIR /app

### CLI only
FROM base AS light

COPY --from=build /app/bin/local-inference /app/bin/local-inference-cli /app/

ENTRYPOINT [ "/app/local-inference-cli" ]

### Server, the default target
FROM base AS server

ENV LOCAL_INFERENCE_ARG_HOST=0.0.0.0

COPY --from=build /app/bin/local-inference /app/bin/local-inference-server /app/

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/local-inference-server" ]
