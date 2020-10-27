
AIDO_REGISTRY ?= docker.io
PIP_INDEX_URL ?= https://pypi.org/simple

build_options=\
 	--build-arg AIDO_REGISTRY=$(AIDO_REGISTRY)\
 	--build-arg PIP_INDEX_URL=$(PIP_INDEX_URL)\
	$(shell aido-labels)



repo=challenge-aido_lf-experiment_manager
# repo=$(shell basename -s .git `git config --get remote.origin.url`)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=$(AIDO_REGISTRY)/duckietown/$(repo):$(branch)

build:
	docker build --pull -t $(tag) $(build_options) .

build-no-cache:
	docker build --pull -t $(tag) $(build_options)  --no-cache .

push: build
	docker push $(tag)
