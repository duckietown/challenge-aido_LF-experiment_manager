

B=dts build_utils
B=dt-build_utils-cli


build:
	$(B) aido-container-build --use-branch daffy \
		--use-org duckietown-infrastructure \
		--push  --ignore-untagged --ignore-dirty  \
		--buildx --platforms linux/amd64,linux/arm64


upload:
	rm -f dist/*
	rm -rf src/*.egg-info
	python3 setup.py sdist
	twine upload --skip-existing --verbose dist/*


bump: # v2

	bumpversion patch
	git push --tags
	git push
