# aika

## Introduction

Aika is a project born out of the desire to make working with time series, and in particular doing time series
research, as painless as possible. As a rule time series computations are always in some sense a 
[DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph), however, when doing such computations "live" we generally
have a complex graph that we want to run up to "now". To do so brings in many quite complex requirements:
* Incremental running - we generally do not want to run the entire history again, so we want to identify how much to run.
* Look back: When running incrementally , we may need data from a considerable lookback on the inputs to calculate a new entry right now, such as eg exponential moving average.
* Completeness: Data may not be available for some inputs at the expected time. In that case, if we simply run all the nodes we may incorrectly evaluate new rows,
when the desired behaviour should be to wait until that data is ready. 

This requirements represent quite complex use cases, and are vital to the correct running of timeseries systems, however,
they are also hard to get right. There are also further requirements that such a system might require: distributed computing
is one example, a graph with thousands of nodes in somewhat parallel pipelines will benefit from parallelisation, but this
requires understanding when parents are complete.

The goal of aika is as far as possible to abstract away these three concerns, and make it possible for researchers to
think only of writing simple python functions, running them, and little by little building up a reliable graph that can be
easily transferred into a production setting.

## Notebooks
There are as part of the project some notebooks that will introduce you to the essential parts of
aika. In particular how to create tasks, chain them together, run them and view their outputs. There
is at this time no exhaustive user guide beyond these notebooks. To access these notebooks simply clone
the repository and install the project-level requirements file. This should give a working distribution on
most systems. 

# Developer Notes

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

Note that this requirements file is intended only for the purposes of eg: running the tests in a consistent way.
This code is intended to be used as a collection of libraries and any actual restrictions on versions must be specified
in the setup.py of the individual libraries in the usual ways. This is only a convenience for developers.

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

We will add further parameter types in the future, including: sets and arbitrary hashable python objects.