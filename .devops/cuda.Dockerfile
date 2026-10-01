ARG UBUNTU_VERSION=24.04
# CUDA 12.8 is the oldest release that builds for Blackwell; the host driver must be 570 or newer
ARG CUDA_VERSION=12.8.1
ARG GCC_VERSION=14
ARG NODE_VERSION=24

ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

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
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION
# Turing (RTX 20xx, T4) to Blackwell (RTX 50xx); 120a is the architecture-specific Blackwell target
ARG CUDA_DOCKER_ARCH="75-real;80-real;86-real;89-real;90-real;120a-real"

RUN apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake git libssl-dev libgomp1

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# the CPU backend is built for every x86 variant and the one the host supports is loaded at run time
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j$(nproc) --target local-inference-server local-inference-cli local-inference-app

RUN mkdir -p /app/lib /app/bin && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    cp build/bin/local-inference-server build/bin/local-inference-cli build/bin/local-inference /app/bin

### Run time
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

ARG BUILD_DATE
ARG APP_VERSION
ARG APP_REVISION
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="inmoloc" \
      org.opencontainers.image.description="Concurrent LLM serving on your own GPU" \
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
