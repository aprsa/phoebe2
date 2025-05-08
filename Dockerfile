FROM python:3-alpine

ARG PHOEBE_BRANCH=master

RUN apk add --no-cache \
      git \
      build-base \
      musl-dev

RUN git clone --branch ${PHOEBE_BRANCH} --single-branch --depth 1 \
      https://github.com/phoebe-project/phoebe2.git /phoebe

WORKDIR /phoebe

RUN pip install --upgrade pip

RUN pip install --no-cache-dir .

RUN mkdir -p /workspace
WORKDIR /workspace

ENTRYPOINT ["python", "-c", "import sys, phoebe; print(f'Python {sys.version}\\nPHOEBE {phoebe.__version__}')"]
