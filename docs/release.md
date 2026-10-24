# PyPI release

## Setup

Put a PyPI API token in `~/.pypirc`. A section with a name other than `pypi` needs the `repository` line, or twine fails with `KeyError: 'repository'`.

```
[pypi-modelpipe]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = <token>
```

## Release

Set `version` in `pyproject.toml`, then:

```
uv lock
git commit -am "bump version to X.Y.Z"
git tag -a vX.Y.Z -m vX.Y.Z
git push origin master vX.Y.Z
rm -rf dist
uv build
uvx twine check dist/*
uvx twine upload -r pypi-modelpipe dist/*
```

`uv lock` records the new version in `uv.lock`.

Use `uvx twine`. Hatchling writes metadata version 2.5, and twine 6.2 rejects it.

The wheel should have one top-level directory, `modelpipe/`, with the elements under `modelpipe/plugins/python`.
