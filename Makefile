

B=dts build_utils
B=dt-build_utils-cli


build:
	$(B) aido-container-build --use-branch daffy \
		--use-org duckietown-infrastructure \
		--push  --ignore-untagged --ignore-dirty  \
		--buildx --platforms linux/amd64,linux/arm64

upload: # v3
	dts build_utils check-not-dirty
	dts build_utils check-tagged
	dts build_utils check-need-upload --package duckietown-experiment-manager-daffy make upload-do

upload-do:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*


bump: # v2

	bumpversion patch
	git push --tags
	git push
