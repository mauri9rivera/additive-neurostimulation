# setup.py
from setuptools import setup, find_packages

def read_requirements(fname="requirements.txt"):
    with open(fname) as f:
        reqs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return reqs

setup(
    name="additive_neurostim",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    include_package_data=True,
)