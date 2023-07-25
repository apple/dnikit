components := dnikit dnikit_tensorflow dnikit_torch

clean_dirs := src/dnikit*/dist junit*.xml coverage .coverage* .env* docs/_build
clean_dirs += */.pytest_cache src/dnikit*/.pytest_cache .pytest_cache .mypy_cache

export PIP_INDEX_URL := https://pypi.org/simple

################
# INSTALLATION #
################
all: install

install: cmd = install -s --deps=develop --extras=notebook,image,dimreduction,dataset-report,tf2
install: $(components)
	@echo "Done installing DNIKit."

install-tf1-gpu: cmd = install -s --deps=develop --extras=notebook,image,dimreduction,dataset-report,tf1-gpu
install-tf1-gpu: $(components)
	@echo "Done installing DNIKit with TF1 GPU."

install-tf1: cmd = install -s --deps=develop --extras=notebook,image,dimreduction,dataset-report,tf1
install-tf1: $(components)
	@echo "Done installing DNIKit with TF 1."

$(components):
	@pip install -U flit$(FLIT_VER) flit_core$(FLIT_VER)
	@flit -f src/$@/pyproject.toml $(cmd)

uninstall:
	@pip uninstall --yes $(components)

########
# TEST #
########
test: test-pytest test-notebooks
	@echo "Done running tests."

test-all: test-pytest-all test-notebooks
	@echo "Done running tests."

test-pytest:
	@echo "Running pytest."
	@pytest

test-pytest-all:
    # runs all tests including ones marked @pytest.mark.slow (see conftest.py)
    # this is typically triggered via the nightly builds on rio
	@echo "Running pytest (all)."
	@pytest --runslow --durations=10

test-smoke:
	@echo "Running pytest smoke tests."
	@pytest -m "not regression"

test-notebooks:
	@echo "Converting notebooks"
	@jupyter nbconvert --to python --output-dir notebooks/.verify notebooks/*/*.ipynb
	@echo "Check notebooks"
	@mypy notebooks/.verify


#########
# OTHER #
#########
doc:
	@$(MAKE) -C docs html
	@echo "\nOpen docs/_build/html/index.html to get started.\n"

clean:
	@-rm -rf $(clean_dirs)


.PHONY: all install install-tf1-gpu install-tf1 uninstall $(components) test test-all test-pytest test-pytest-all test-smoke test-notebooks doc clean
