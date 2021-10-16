ARG ARCH=amd64
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}

ARG DOCKER_REGISTRY
FROM ${DOCKER_REGISTRY}/duckietown/aido-base-python3:${BASE_TAG}


ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}


RUN wget -q https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
    tar xvf ffmpeg-git-amd64-static.tar.xz --strip-components=1
RUN cp ./ffmpeg /usr/bin/ffmpeg
RUN which ffmpeg
RUN ffmpeg -version


COPY requirements.pin.txt ./
RUN python3 -m pip install  -r requirements.pin.txt

COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install  -r .requirements.txt
RUN python3 -m pip uninstall -y dataclasses
RUN python3 -m pip list
RUN pipdeptree



COPY . .

RUN python3 -m pip install . --no-deps
RUN python3 -m pip list

RUN dt-experiment-manager --help
RUN ls -al *png
ENTRYPOINT dt-experiment-manager
