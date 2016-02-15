Cortical microcircuit simulation

 Multi-layer microcircuit model of early sensory cortex
(Potjans, T. C., & Diesmann, M. (2014) Cerebral Cortex
24(3):785-806, code available at
http://www.opensourcebrain.org/projects/potjansdiesmann2014),
originally implemented in NEST
(http://nest-simulator.org). PyNN version modified to run
as task in the Collaboratory. Simulation parameters are
defined in microcircuit.yaml, which needs to be passed as
a configuration file. A template can be downloaded from
https://github.com/INM-6/UP-Tasks. It is possible to
provide an empty or partial configuration file. For the
missing parameters, default values will be used. After
uploading the YAML file, its content type needs to be
changed to
'application/vnd.juelich.simulation.config'. Parameters
defined in the WUI overwrite values defined in the
configuration file. For running the full model, 8 CPU
cores and 15360 MB memory should be requested.

Contributors:
    NEST Developers
    Johanna Senk (j.senk@fz-juelich.de)
    Sacha van Albada

Instructions:
    Running this task locally:
    - install task_sdk from https://developer.humanbrainproject.eu/docs/projects
      /task-sdk/0.0.21/index.html
    - python microcircuit_task.py

Output:
    - returns recorded data as defined in the configuration file, e.g., spikes,
      membrane voltages, or correlation functions

Tested versions on the Collaboratory:
    NEST: 2.6.0
    Python: 2.6.6
    NumPy: 1.8.0
    PyNN: 0.7.5
