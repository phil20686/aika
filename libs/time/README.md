# Introduction
aika time is part of the aika project for working with time series data, you can find more details
on the [project webpage](https://github.com/phil20686/aika/): 

# aika-time

Utility methods for working with timestamps and time series in the context of pandas. Principal components are:

1. Time stamp class that enforces olsen timestamps, i.e. all timestamps must have a time zone.
2. TimeOfDay class that represents a time of day in a given timezone, and given a 
set of dates produce the correct timestamp.
3. TimeRange class that represents a timerange.
4. causal.py - utilities for aligning one series on another in a causally correct way.

# Install
```python -m pip install aika-time```


