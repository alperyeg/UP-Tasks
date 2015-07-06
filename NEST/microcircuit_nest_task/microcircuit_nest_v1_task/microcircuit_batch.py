# Sacha van Albada, Apr 2015

# Run the microcircuit model for different random seeds,
# both for the connectivity and for the Poisson drive

import os
import numpy
import random
from sim_params import *

nodes = system_params['n_nodes'] # how many compute nodes to use
procs_per_node = system_params['n_procs_per_node'] # how many MPI processes to use per compute node
procs = nodes * procs_per_node # how many processor cores to use in total
mem = system_params['memory']  # how much total memory required (in GB)

simname = 'microcircuit_batch'

# where your python code for the microcircuit model resides
workingdir = '/users/albada/Documents/spinnaker/PyNN'
# output base directory
output_dir = '/scratch/albada/spinnaker/PyNN/'

# python file to be executed by the queue
pyf_name = 'microcircuit.py'

# job description file
jdf_name = 'microcircuit_batch.jdf'

nsims = 1
# master seeds for NEST Poisson generators
msd = random.sample(xrange(500000, 10000000), nsims)
# seeds for V and connectivity
pyseeds = random.sample(xrange(500000, 10000000), nsims)

for i in xrange(nsims):

    # output directory for this parameter combination
    this_output_dir = '48MPIprocs'#str(i)
    full_output_dir = output_dir + '/' + this_output_dir
    # create directory if it doesn't exist yet
    if this_output_dir not in os.listdir(output_dir):
        os.system('mkdir ' + full_output_dir)

    # file into which error/stdout is written
    qsubfile = full_output_dir + '/qsub'
                
    # make directory 'PyNN' in output directory if it doesn't exist yet
    if 'PyNN' not in os.listdir(full_output_dir):
        os.system('mkdir ' + full_output_dir + '/PyNN')

    os.chdir(workingdir)

    # copy all the relevant files to the output directory
    os.system('cp microcircuit.py ' + full_output_dir + '/PyNN')
    os.system('cp network.py ' + full_output_dir + '/PyNN')
    os.system('cp network_params.py ' + full_output_dir + '/PyNN')
    os.system('cp sim_params.py ' + full_output_dir + '/PyNN')
    os.system('cp get_conductances.py ' + full_output_dir + '/PyNN')
    os.system('cp helper_functions.py ' + full_output_dir + '/PyNN')
    os.system('cp scaling.py ' + full_output_dir + '/PyNN')
    os.system('cp connectivity.py ' + full_output_dir + '/PyNN')
    os.system('cp plotting.py ' + full_output_dir + '/PyNN')

    os.chdir(full_output_dir + '/PyNN')

    # write custom parameter files, which will be invoked in the simulation after importing the default parameters
    # Any two master seeds must differ by at least 2*n_vp + 1 to avoid correlations between simulations.
    f = open(full_output_dir + '/PyNN/custom_network_params.py', 'w')
    f.write("master_seed = " + str(msd[i]) + '\n')
    f.write("pyseed = " + str(pyseeds[i]) + '\n')
    f.close()

    f = open(full_output_dir + '/PyNN/custom_sim_params.py', 'w')
    f.write("system_params['output_path'] = '" + full_output_dir + "'")
    f.close()
 
    this_pyf_name = full_output_dir  + '/PyNN/' + pyf_name

    # write job description file
    f = open(full_output_dir + '/PyNN/' + jdf_name , 'w')
    f.write( '#!/bin/bash \n')
    # set job name
    f.write( '#PBS -N '+simname+'\n' )
    # working directory
    f.write( '#PBS -d '+full_output_dir+'/PyNN\n' )
    # output file to send standard/error output stream to
    f.write( '#PBS -o '+qsubfile+'\n' )
    # join error stream and output stream to obtain a single file in the end
    f.write( '#PBS -j oe \n' )
    # request a total of nodes*processes_per_node for this job
    f.write('#PBS -l nodes=%i:ppn=%i\n' % (nodes, procs_per_node))
    # request processor time
    f.write( '#PBS -l walltime=8:00:00\n' )
    # request mem MB of memory
    f.write( '#PBS -l mem=' + mem + ' \n' )
    f.write('export PYTHONPATH=' + system_params['backend_path'] + ':' + system_params['pyNN_path'] + '\n')
    f.write('. /usr/local/mpi/openmpi/1.4.3/gcc64/bin/mpivars_openmpi-1.4.3_gcc64.sh\n')
# To reduce the build time (perhaps at the expense of simulation time), preload a different allocator as follows:
#    f.write('mpirun -x LD_PRELOAD=/usr/local/gperftools/lib/libtcmalloc_minimal.so.4.1.0 -np %i python %s\n' % (procs, this_pyf_name))
    f.write('mpirun -np %i python %s\n' % (procs, this_pyf_name))
    f.close()

    # submit job
    os.system('qsub ' + jdf_name)

