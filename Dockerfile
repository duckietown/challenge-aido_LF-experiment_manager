ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy-amd64

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN pip3 install --use-feature=2020-resolver -r .requirements.txt
RUN pip3 uninstall -y dataclasses
RUN pip3 list
RUN pipdeptree



COPY . .

RUN pip install . --no-deps
RUN pip3 list

RUN  dt-experiment-manager --help

RUN wget -q https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
    tar xvf ffmpeg-git-amd64-static.tar.xz --strip-components=1
RUN cp ./ffmpeg /usr/bin/ffmpeg
RUN which ffmpeg
RUN ffmpeg -version
#RUN apt-get update && apt-get install -y  software-properties-common && \
#    add-apt-repository ppa:jonathonf/ffmpeg-4 && \
#    apt-get update && apt-get install -y ffmpeg
#RUN ffmpeg -version
RUN cat /etc/issue

ENTRYPOINT dt-experiment-manager
