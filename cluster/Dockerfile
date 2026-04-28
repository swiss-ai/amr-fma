FROM nvcr.io/nvidia/pytorch:26.02-py3

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

ARG BRANCH=main

WORKDIR /workspace

COPY constraints-packages.txt /workspace/constraints-packages.txt
RUN REGEX=$(paste -sd'|' /workspace/constraints-packages.txt | sed 's/^/^(/;s/$/)==?/') && \
    pip list --format=freeze | grep -E "$REGEX" > /workspace/frozen.txt

RUN git clone --branch ${BRANCH} --single-branch https://github.com/swiss-ai/amr-fma.git
WORKDIR /workspace/amr-fma

RUN uv pip install --system --break-system-packages \
    . \
    --constraint /workspace/frozen.txt

ENV CUDA_CACHE_DISABLE=1

CMD ["/bin/bash"]
