# Introduction

The aika project is a collection of libraries for working with time series data, and in particular with
timeseries data that is expected to be continuously updated. The datagraph sub project is a way to store
time series data along with any parameters used to generate that data including other data. It comes with
several different persistence engines which can be used to browse stored data and filter by parameters. Since
it embeds the dependency graph it enforces consistency. Importantly, meta data such as the time range of
the data is stored in the meta data allowing consumers to check the existence of data without examining it.
Within the aika project this is used as the backing layer to the task library (putki) which allows the tasks
to efficiently know which tasks need to be run at any given point in time.

You can read more about the aika project on [the project webpage](https://github.com/phil20686/aika/).

# Installation
``python -m pip install aika-datagraph``