# LOFAR VLBI pipeline

The Low-Frequency Array Very Long Baseline Interferometry (LOFAR-VLBI) pipeline is a calibration and imaging pipeline that includes all of LOFAR’s international stations.
This is an implementation of the LOFAR-VLBI pipeline in the Common Workflow Language, which will eventually replace the [Genericpipeline](https://github.com/lmorabit/lofar-vlbi) implementation.

Instruction on setting up, configuration and usage of the pipeline can be found in the dedicated [wiki](https://git.astron.nl/RD/VLBI-cwl/-/wikis/home).

## Installing the pipeline

The simplest way to get set up is to install the package in a virtual environment, _e.g._ by running the following from the repository root directory:
```
python -m venv venv
. venv/bin/activate
pip install .
```


## Running the test suite

The pipeline comes with a test suite.
This suite can be run from the command line by installing the testing dependencies.
It is recommended to do this from a virtual environment:
```
python -m venv venv
. venv/bin/activate
pip install .[test]
```
The LOFAR VLBI pipeline uses [tox](https://tox.wiki/) to run its testing suite, which is available in the virtual environment and can be run from the project root directory.
By default, tox tests against Python versions 3.9 through 3.13 (if available).
To test against a specific version, run tox with the `-e` flag.
For example, to test against Python 3.10, run
```
tox -e py310
```
The testing suite can also be run using [`pytest`](pytest.org).
In this case, the variable `VLBI_ROOT_DIR` must be set to the project root directory and `PYTHONPATH` must include the project's `scripts` directory.
For example, if run from the project root directory:
```
VLBI_ROOT_DIR=$PWD PYTHONPATH=$PWD/scripts pytest
```
