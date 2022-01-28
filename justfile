setup:
    git submodule update --init
    poetry install
    poetry run python setup.py develop

setup-cu11:
    @just setup
    poetry run pip install --upgrade \
        torch==`poetry export | grep torch== | cut -d';' -f1 | cut -d'=' -f3`+cu113 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html
    poetry run python setup.py develop

docker-dev:
    docker build -f docker/Dockerfile.dev .

build:
    poetry run python setup.py develop

test: build
    poetry run pytest

lint:
    poetry run isort --check rmsnorm tests
    poetry run black --check --include .py --exclude ".pyc|.pyi|.so" rmsnorm tests
    poetry run black --check --pyi --include .pyi --exclude ".pyc|.py|.so" rmsnorm tests
    poetry run pylint rmsnorm
    poetry run pyright rmsnorm tests

fix:
   poetry run isort rmsnorm tests
   poetry run black --include .py --exclude ".pyc|.pyi|.so" rmsnorm tests
   poetry run black --pyi --include .pyi --exclude ".pyc|.py|.so" rmsnorm tests
