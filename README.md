# aika

## Repo structure

All libraries should be defined in their own subfolder under the `libs/` directory

Dependencies for each package should be declared inline in the `install_requires` list
of `setup.py`

Note that because we are creating different packages and distributions under the same namespace,
there must be no `__init__` under src/aika, but only under src/aika/package should have the first init,
otherwise the modules will not be importable. See [python documentation](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/) on namespace packages.

## Compiling dependencies

The `requirements.txt` file at the top level contains all the dependencies for all
the packages in the repo. 

This file can be updated by running the `compile_requirements.sh` script. This should be
done if any of the following apply:

- You have created a new library under `libs/`
- You have added, removed, or changed the version specified of any dependency on an existing library
- You wish to update dependencies to pick up recent releases
