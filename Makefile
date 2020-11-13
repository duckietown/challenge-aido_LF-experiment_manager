

build:
	dts build_utils aido-container-build --use-branch daffy --ignore-untagged --ignore-dirty


push: build
	dts build_utils aido-container-push  --use-branch daffy


bump: # v2
	bumpversion patch
	git push --tags
	git push
