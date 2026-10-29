# Nsight Systems on its own: built once (its base never changes, so the install stays cached), copied into the
# nsight-systems volume by profile-server.sh, and mounted into whichever server image is profiled
FROM docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nsight-systems-2026.3.2 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sfn "$(ls -d /opt/nvidia/nsight-systems/*/ | head -1)" /opt/nvidia/nsight-systems/current && \
    /opt/nvidia/nsight-systems/current/bin/nsys --version
