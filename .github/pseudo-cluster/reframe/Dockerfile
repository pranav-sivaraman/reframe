# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

# HyperQueue is built from source because the arm64 release binaries need
# glibc 2.39, but the cluster images are Ubuntu 22.04 (glibc 2.35). Building
# on the same base keeps the binary compatible on both amd64 and arm64.
#
# Build stage adapted from https://docs.docker.com/guides/rust/
ARG RUST_VERSION=1.95
FROM docker.io/library/rust:${RUST_VERSION}-slim AS rust

FROM ubuntu:22.04 AS hq-build

# The rustup toolchain only needs an old glibc, so it runs on Ubuntu 22.04
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY --from=rust /usr/local/cargo /usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# TODO: switch to the release source tarball once a release includes
# https://github.com/It4innovations/hyperqueue/pull/1170
ARG HQ_URL=https://github.com/It4innovations/hyperqueue/archive/2a05d6da37f5bd94e1feadbb00e584d6731adb43.tar.gz
ARG HQ_SHA256=b51ad6a42445df8f68d9087fb97f63414578452e4dc5c49ba139491e62b42ad8

WORKDIR /app

# cmake and libclang are needed to build the bundled HiGHS solver
ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
set -e
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates clang cmake libclang-dev
rm -rf /var/lib/apt/lists/*
EOF

ADD --checksum=sha256:${HQ_SHA256} ${HQ_URL} /tmp/hq.tar.gz

# Cache mounts keep the multi-GB build tree out of the image layers; copy the
# binary out before /app/target is unmounted
RUN --mount=type=cache,target=/app/target/ \
    --mount=type=cache,target=/var/cache/cargo \
    <<EOF
set -e
tar -xzf /tmp/hq.tar.gz --strip-components=1
CARGO_HOME=/var/cache/cargo cargo build --locked --release -p hyperqueue
cp ./target/release/hq /bin/hq
EOF

FROM ubuntu:22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install -y \
    build-essential \
    clang jq libomp-dev tree vim \
    git \
    mariadb-client \
    munge \
    slurm-client \
    slurm-wlm-torque \
    sudo \
    python3 \
    python3-pip \
    wget \
    curl \
    mpich \
    libmpich-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m admin -s /usr/bin/bash -d /home/admin && \
    echo "admin:admin" | chpasswd && adduser admin sudo && \
    echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir /scratch && \
    chown -R admin:admin /scratch

# The entrypoint copies this to the shared home, so that the compute nodes
# find `hq` in the same path as the login node
COPY --from=hq-build /bin/hq /usr/local/share/hq/hq

COPY .github/pseudo-cluster/reframe/docker-entrypoint.sh /usr/bin/

RUN chmod +x /usr/bin/docker-entrypoint.sh

WORKDIR /home/admin

COPY . /usr/local/share/reframe

ENV USER admin
ENV SHELL bash

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
