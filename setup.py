from setuptools import setup


def get_version(filename: str):
    import ast

    version = None
    with open(filename) as f:
        for l in f:
            if l.startswith("__version__"):
                version = ast.parse(l).body[0].value.s
                break
        else:
            raise ValueError(f"No version found in {filename!r}.")
    if version is None:
        raise ValueError(filename)
    return version


fversion = get_version(filename="src/duckietown_experiment_manager/__init__.py")

line = "daffy"
install_requires = [
    # f"duckietown-gym-{line}",
    # f"aido-agents-{line}",
    "multidict",
    "aiohttp",
    # "py-multihash",
    # "py-cid",
    "pytz",
    "zuper-nodes-z6",
    "zuper-ipce-z6",
    "zuper-typing-z6",
    "numpy",
    "PyYAML",
    f"aido-protocols-{line}",
    f"aido-analyze-{line}",
    f"duckietown-world-{line}",
    f"duckietown-challenges-{line}",
    "procgraph-z6",
    # "PyGeometry-z6",
]

setup(
    name=f"duckietown-experiment-manager-{line}",
    version=fversion,
    keywords="",
    package_dir={"": "src"},
    packages=["duckietown_experiment_manager"],
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["dt-experiment-manager = duckietown_experiment_manager.experiment_manager:go",],
    },
)
