ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy-amd64

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}


RUN wget -q https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
    tar xvf ffmpeg-git-amd64-static.tar.xz --strip-components=1
RUN cp ./ffmpeg /usr/bin/ffmpeg
RUN which ffmpeg
RUN ffmpeg -version

RUN pip3 install -U "pip>=20.2"

COPY requirements.pin.txt ./
RUN pip3 install  -r requirements.pin.txt

COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN pip3 install  -r .requirements.txt
RUN pip3 uninstall -y dataclasses
RUN pip3 list
RUN pipdeptree



COPY . .

RUN pip install . --no-deps
RUN pip3 list

RUN dt-experiment-manager --help
RUN ls -al *png
ENTRYPOINT dt-experiment-manager
