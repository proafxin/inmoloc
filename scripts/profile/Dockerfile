# adds Nsight Systems to a server image: docker build --build-arg BASE=<image> -t <image>-nsys .
# both server images (inmoloc and llama.cpp) already have NVIDIA's CUDA apt repository, which carries Nsight Systems
ARG BASE=inmoloc:latest
FROM ${BASE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2026.3.2 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf "$(ls -d /opt/nvidia/nsight-systems/*/bin | head -1)/nsys" /usr/local/bin/nsys && nsys --version
