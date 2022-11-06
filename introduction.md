
# Aika - User's Introduction

## Introduction

Aika exists to make research and production of time-series as easy as 
possible. The generic problem that aika exists to solve is one like the following,
you get updates, say, market prices, at regular intervals, say, 15 minute bars,
and you would like to update a downstream dataseries, say, a rolling average, 
every time you have a new 15 minute bar. Yet other tasks may take that as input. 

There are two approaches to this, one is event driven, you produce a series of nodes,
that fire when they get a new message, and pass their result on, and so forth. While
excellent message based systems like eg, rabbit mq exist, they have, in this authors
opinion, quite a high barrier to entry. Moreover, writing code that does incremental updates
on things is inherently quite hard, and naturally introduces quite hard state dependence
problems. If one message service errors, downstream services can quite happily give
incorrect answers treating the last valid message from a broken system as if it is up
to date. This also produces a series of quite hard problems.

The other approach is that taken by aika, we adopt a model in which, at all times, the
data is the state. Persistence engines store complete timeseries and append to them. Tasks
can be given an expectations around the time series index about what it means to be complete,
and if they are incomplete this will prevent the updates of any downstream tasks. Thus,
in the many common tasks where the data is known to be regular in time, we can know that
any data that is computed, will be correct, since tasks will not run if their predecessors
outputs do not meet their expectations for completeness.

Further to this, datasets know their parameterisations, including the parameterisations of their
predecessors, so if you change how a task is parameterised, the datasets below it will know whether
they need to recompute.

## Basic Concepts

### Dataset
A dataset is represented by the aika.datagraph.interface.DataSet. A dataset consists of
some kind of data, which should be a time-indexed pandas Tensor (a dataframe or series)
along with a metadata object.

### DataSetMetadata & DataSetMetadataStub
Dataset metadata represents all the data required to uniquely define a dataset. Two datasets
should have equal metadata if and only if they are the same data. Dataset metadata combines three
different kinds of data: 
* Data about how a dataset is computed - function arguments, input data
* Data about how a dataset is stored - the persistence engine used to store it for example
* Data about its semantic meaning - a name, the code version used to compute it etc.

A DataSetMetadata points directly to its predecessor metadata, forming a graph that can be walked,
a DataSetMetadataStub differs only in that its predecessors will only be stubs. Their data will be
lazily fetched from the persistence engine if it is required, but since walking the full graph can
be expensive this allows us to fetch only the data that we need when eg, pulling a dataset.

### PersistenceEngines
Persistence engines store datasets, they also allow you to modify the data attached to a dataset, such
as appending new data (only writes lines after the last existing data) or merging (essentially the same
as pd.DataFrame.combine_first), aswell as inserts, replacement and deletion. They further allow
queries on existing datasets, such as finding datasets with particular names or parameterisations.

### Tasks
Tasks are the workhorse of how we compute data, they are also, in the main, entirely abstracted
from the user. Each task has, in addition to its functional logic, an output, a set of dependencies, a completion checker, and a target time range.

#### Outputs
An output is just the DataSetMetadata of the expected output of running the task. This can exist whether
or not the data it refers to exists yet.

#### Target time range
This is the data that you want the task to compute.

#### Completion Checker
A completion checker is, in essence, just a function that can inspect the time range spanned by
any existing data defined by the output, and then can evaluate whether the task needs to run again.
Generally this is as simple as the checker having a rule such as "there must be row within x
minutes of the end of the target timerange". Completion checkers only evaluate the end of the
data, if you expand the data time range at the start, that is essentially a semantic error. 
Causally correct time series data can never be guaranteed if a predecessor suddenly has more
data into the past. To learn more about completion checking, see the notebook "Completion Checking Example".


#### Dependencies
Dependencies are in essence simply the predecessor datasets from the output metadata, however, a task,
in addition to knowing what datasets it needs to point to, also needs to know how much data it needs.
Suppose I want to compute the 100 day moving average for daily data in the target_timerange from "2000-01-01"
to "2000-01-05"), to do that, the data I need to pull from a dependency is going to be from "1999-9-23"
to "2000-01-05". Dependencies will usually be created automatically by the context from another task,
but in rare cases you need to wrap them in a dependency so that you can provide lookbacks for functions
that need past history to correctly compute small increments. A dependency wrapper also
allows you to specify whether to inherit a completion checker. In general, if you do not
specify a completion checker, the completion checker of a task will be the strictest completion
check among all predecessor tasks - this is equivalent to assuming that the last row of the output
is expected to be the latest among the last rows of the inputs. However, you can use dependency 
wrappers around a task to control which completion checker is used.

### Context

Contexts create tasks, while removing as much as possible of the boilerplate. Most of the
time, for example, usually you want a single persistent engines in all your tasks. So the
context will insert its default persistence engine in, saving you the effort. Similarly,
a context will create a time series task, by inspecting a python function and the arguments
and creating a task with the appropriate output. To learn more about contexts, see
the notebook "Example MACD Signal"

### Runners
A runner runs a graph. Every task in the graph is, by construction, a pickleable function
that knows itself how to extract its data from its parent persistence engines. This means
that a runner is really just a trivial orchestrator that makes sure that tasks run in the right
order, and produces nice looking error messages if things go wrong. Given a set of target
tasks, a runner will walk back through their parents, checking completeness, and running those
that are not yet complete.


