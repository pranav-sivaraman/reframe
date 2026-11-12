#!/bin/bash

trap exit 0 INT

sudo service munge start

# Install HyperQueue in the shared home directory, so that it is available
# in the same path in the compute nodes; `.profile` adds it to the PATH
mkdir -p $HOME/.local/bin
cp /usr/local/share/hq/hq $HOME/.local/bin/hq

# Needs to be copied in the shared home directory
cp -r /usr/local/share/reframe .
cd reframe
uv sync --group dev
source $HOME/.profile

echo "Running unittests with backend scheduler: ${BACKEND}"

tempdir=$(mktemp -d -p /scratch)
TMPDIR=$tempdir uv run coverage run --source=reframe ./test_reframe.py \
    --rfm-user-config=ci-scripts/configs/ci-cluster.py \
    --rfm-user-system=pseudo-cluster:compute-${BACKEND:-squeue}
uv run coverage xml -o coverage.xml
