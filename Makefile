# FILE: Makefile
# -------------------------------------------------------------------------------------------------
SHELL := /bin/bash
.PHONY: setup lint test clean kaggle-export package data-pull

setup:
\t@echo '[WDE] installing deps (all extras) & pre-commit hooks'
\tpoetry install --with dev,geo,ml,viz,notebook
\tpoetry run pre-commit install

lint:
\tpoetry run pre-commit run --all-files || true

test:
\tpoetry run pytest

data-pull:
\t@echo '[WDE] placeholder: implement data pulls in src/wde/data/ and wire here'
\t@echo 'Tip: keep large data in DVC and document sources in datasets.md'

kaggle-export:
\t@echo '[WDE] exporting Kaggle notebook…'
\t@mkdir -p notebooks
\t@# Example: convert a script/template to notebook (adjust to your flow):
\t@# poetry run jupyter nbconvert --to notebook --execute templates/wde_kaggle_template.ipynb \\
\t@#   --output notebooks/wde_kaggle.ipynb
\t@echo 'NOTE: Add templates/ + src/wde to enable real export.'

package:
\t@echo '[WDE] packaging release artifacts…'
\t@mkdir -p artifacts
\t@tar -czf artifacts/wde_$(shell date +%Y%m%d)_bundle.tgz README.md LICENSE CITATION.cff

clean:
\trm -rf .pytest_cache .ruff_cache __pycache__ **/__pycache__ .mypy_cache
\tfind . -name '*.pyc' -delete
\tfind . -name '*.pyo' -delete
\tfind . -name '*.log' -delete