import os
from pathlib import Path

def test_repo_paths_and_imports():
    # Verify notebooks directory exists
    nb_dir = Path('notebooks')
    assert nb_dir.exists(), 'notebooks/ directory missing'

    # Sanity check: at least one notebook present
    nbs = list(nb_dir.glob('*.ipynb'))
    assert len(nbs) >= 1, 'No notebooks found in notebooks/'
    
    # Attempt to import core package
    try:
        import world_engine  # noqa: F401
    except Exception as e:
        raise AssertionError(f'Failed to import world_engine: {e}')

    # Validate that we can convert a notebook to HTML (parse only, not execute)
    try:
        import nbformat
        from nbconvert import HTMLExporter
        from traitlets.config import Config

        c = Config()
        exporter = HTMLExporter(config=c)

        with nbs[0].open('r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        (body, resources) = exporter.from_notebook_node(nb)
        assert '<html' in body.lower(), 'Notebook did not convert to HTML'
    except Exception as e:
        raise AssertionError(f'Notebook convert check failed: {e}')