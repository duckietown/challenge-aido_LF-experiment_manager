ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip3 install -U pip>=20.2
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip3 list
RUN pipdeptree


COPY . .

RUN  python3 -c "from  experiment_manager import *"

ENTRYPOINT ["python3", "experiment_manager.py"]
