###################################################
###     	Simulation parameters		###        
###################################################

import os

simulator_params = {
    'nest':
    {
      'timestep'        : 0.1,
      'threads'         : 1,
      'sim_duration'    : 1000.,
      'min_delay'       : 0.1,
      'max_delay'       : 100.
      # When max_delay is not set, FixedTotalNumberConnect sometimes
      # produces an error as requested delays are larger than the default 10 ms
      # Setting max_delay to np.inf is not good either: the simulation
      # fails as buffers are probably too large
    }
}

system_params = {
    # number of nodes
    'n_nodes'           : 1,
    # number of MPI processes per node
    'n_procs_per_node'  : 24,
    # walltime for simulation
    'walltime'          : '8:0:0',
    # total memory for simulation
    'memory'            : '4gb', # For 12 or 24 MPI processes, 4gb is OK. For 48 MPI processes, 8gb doesn't work, 24gb does.
    # file name for standard output
    'outfile'           : 'output.txt',
    # file name for error output
    'errfile'           : 'errors.txt',
    # absolute path to which the output files should be written
    'output_path'       : os.getcwd() + '/output',
    # Directory for connectivity I/O
    'conn_dir'		: os.getcwd() + '/conns',
    # path to the MPI shell script
    'mpi_path'          : '/usr/local/mpi/openmpi/1.4.3/gcc64/bin/mpivars_openmpi-1.4.3_gcc64.sh',
    # path to back-end
    'backend_path'      : '/users/albada/nest-2.6.0_install/lib64/python2.6/site-packages',#'/users/albada/10kproject_10711_install.hambach/lib64/python2.6/site-packages',
    # path to pyNN installation
    'pyNN_path'         : '/users/albada/PyNN-0.7.5_install_hambach/lib/python2.6/site-packages',
    # command for submitting the job
    'submit_cmd'        : 'qsub'
}

# make any changes to the parameters
if 'custom_sim_params.py' in os.listdir('.'):
    execfile('custom_sim_params.py')
