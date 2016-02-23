# @ job_name = "microcircuit_hpc"
# @ comment = "job microcircuit_hpc"
# @ error = /homea/jinb33/jinb3320/github_test/microcircuit_hpc_task/out.txt
# @ output = /homea/jinb33/jinb3320/github_test/microcircuit_hpc_task/out.txt
# @ environment = COPY_ALL
# @ wall_clock_limit = 0:30:00
# @ notification = error
# @ notify_user = j.senk@fz-juelich.de
# @ job_type = bluegene
# @ bg_size = 32
# @ queue

export PYTHONPATH=$HOME/nest_test/lib/python3.4/site-packages/:$PYTHONPATH

module load python3/3.4.2

NESTCODE="/homea/jinb33/jinb3320/github_test/microcircuit_hpc_task/microcircuit.py"

runjob --ranks-per-node 1 --exp-env HOME --exp-env PATH --exp-env LD_LIBRARY_PATH --exp-env PYTHONUNBUFFERED --exp-env PYTHONPATH : /bgsys/local/python3/3.4.2/bin/python3 $NESTCODE