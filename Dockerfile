ARG AIDO_REGISTRY
FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

COPY requirements.* ./
RUN pip install -U -r requirements.resolved
RUN pip list
RUN pipdeptree


COPY . .

RUN  python3 -c "from  experiment_manager import *"

ENTRYPOINT ["python3", "experiment_manager.py"]
