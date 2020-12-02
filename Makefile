

build:
	dts build_utils aido-container-build --use-branch daffy --ignore-untagged --ignore-dirty


push: build
	dts build_utils aido-container-push  --use-branch daffy

upload: # v3
	dts build_utils check-not-dirty
	dts build_utils check-tagged
	dts build_utils check-need-upload --package duckietown-gym-daffy make upload-do

upload-do:
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*


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
