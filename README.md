# aika

## Repo structure

All libraries should be defined in their own sub-folder under the `libs/` directory

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

## User notes

The datagraph currently supports parameters that are one of:
* Python primitives like numbers, strings.
* Tuples
* frozen dicts

For user ease dicts are converted into frozen dicts and lists are converted into tuples upon dataset
creation. That means that a dataset with a list parameter will be considered equal to one with a tuple parameter
with the same values. 

Note that this is a decision about the persistence layer, tasks can be parameterised with lists and tuples, and they
are not converted, this is especially important when working with pandas which treats lists and tuples differently
in indexing operations.

We will add further parameter types in the future, including: sets and arbitrary hashable pythong objects.