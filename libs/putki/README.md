Putki is a task framework designed to make it easy
to build production and research systems on top of timeseries
data. They provide tasks which have a notion of completeness
which is founded on an awareness of what data is expected
from a successful computation, and thus completeness is 
defined via the inspection of parent tasks existing output
and not via knowledge of when a task was last run. This directly
solves many issues around eg mis-computing moving averages due
to unavailable data. 