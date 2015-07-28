Cortical microcircuit simulation: PyNN version modified to run as task on Unified Portal.
Input arguments are defined in microcircuit.yaml and can be modified by the user.

This is an implementation of the multi-layer microcircuit model of early
sensory cortex published by Potjans and Diesmann (2014) The cell-type specific
cortical microcircuit: relating structure and activity in a full-scale spiking
network model. Cerebral Cortex 24 (3): 785-806, doi:10.1093/cercor/bhs358.

Contributors:
    Johanna Senk (j.senk@fz-juelich.de)
    Sacha van Albada
    Long Phan 

Instructions:
    Running this task locally: 
        -> python microcircuit_task.py

    Output:
        -> all output-results in format spikes_xy.dat and one file spiking_activity.png
        (note: adjust file microcircuit.yaml to change the save location of files)

Tested-Version:
    NEST: 2.6.0
    Python: 2.6.6
    PyNN: 0.7.5

