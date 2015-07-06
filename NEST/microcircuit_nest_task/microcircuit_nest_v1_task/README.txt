/*
 *  README.txt
 *
 */

Cortical microcircuit simulation: PyNN version

Contributors:
Sacha van Albada (s.van.albada@fz-juelich.de)
Maximilian Schmidt
Jannis Schücker
Andrew Rowley

This is an implementation of the multi-layer microcircuit model of early
sensory cortex published by Potjans and Diesmann (2014) The cell-type specific
cortical microcircuit: relating structure and activity in a full-scale spiking
network model. Cerebral Cortex 24 (3): 785-806, doi:10.1093/cercor/bhs358.

It has been run on three different back-ends: NEST, SpiNNaker, and the ESS (emulator of HMF)

Files:
	- network_params.py
	Script containing model parameters

        - sim_params.py
        Script containing simulation and system parameters

	- microcircuit.py
	Simulation script

        - network.py
        In which the network is set up

        - connectivity.py
        Definition of connection routines

        - scaling.py
        Functions for adjusting parameters to in-degrees
        to approximately preserve firing rates

        - get_conductances.py
        For determining synaptic conductances, membrane time constants,
        and resting membrane potentials that allow the network with conductance-based
        synapses to closely approximate that with current-based synapses

        - helper_functions.py
        Auxiliary functions for creating the matrix of synaptic weights
        and reading initial membrane potentials from file

	- run_microcircuit.py
	Creates output directory, copies all scripts to this directory, 
        creates sim_script.sh and submits it to the queue
        Takes all parameters from sim_params.sli

	- plotting.py
	Python script for creating raster and firing rate plot


Instructions:

1. Ensure you have the desired back-end.

   For NEST see http://www.nest-initiative.org/index.php/Software:Download
   and to enable full-scale simulation, compile it with MPI support 
   (use the --with-mpi option when configuring) according to the instructions on
   http://www.nest-initiative.org/index.php/Software:Installation

   For using the ESS with VirtualBox, download the BSS Live System from
   http://www.brainscales.org/bss-ubuntu-desktop.iso
   The host system needs to be 64-bit.

   Alternatively, obtain a docker image from the KIP in Heidelberg.

2. Install PyNN 0.7 according to the instructions on 
   http://neuralensemble.org/docs/PyNN/installation.html
   
3. In sim_params.py:

   - set the simulation time via 'sim_duration'
   - set 'output_path' and 'pyNN_path'

   For parallel simulation with NEST further adjust:
   - the number of compute nodes 'n_nodes'
   - the number of processes per node 'n_procs_per_node'
   - queuing system parameters 'walltime' and 'memory'
   - 'mpi_path', 'nest_path'

4. In network_params.py:

   - If not yet present, add dictionary to params_dict for the back-end you wish to use
   - Choose the network size via 'N_scaling' and 'K_scaling', 
     which scale the numbers of neurons and in-degrees, respectively
   - Choose the external input via 'input_type'
   - Optionally activate thalamic input via 'thalamic_input' 
     and set any thalamic input parameters 
   
5. Run the simulation by typing 'python run_microcircuit.py' in your terminal
   (microcircuit.py and the parameter files need to be in the same folder)

6. Output files and basic analysis:
   
   - Spikes are written to .txt files containing IDs of the recorded neurons
     and corresponding spike times in ms.
     Separate files are written out for each population and virtual process.
     File names are formed as 'spikes'+ layer + population + MPI process + .txt
   - Voltages are written to .dat files containing GIDs, times in ms, and the
     corresponding membrane potentials in mV. File names are formed as
     voltmeter label + layer index + population index + spike detector GID +
     virtual process + .dat

   - If 'plot_spiking_activity' is set to True, a raster plot and bar plot
     of the firing rates are created and saved as 'spiking_activity.png'
    

The simulation was successfully tested with NEST revision 10711 and MPI 1.4.3.
Plotting works with Python 2.6.6 including packages numpy 1.3.0,
matplotlib 0.99.1.1, and glob.

---------------------------------------------------

Simulation on a single process:

1. Go to the folder that includes microcircuit.py and the parameter files

2. Adjust 'N_scaling' and 'K_scaling' in network_params.py such that the network
   is small enough to fit on your system 

3. Ensure that the output directory exists, as it is not created via 
   run_microcircuit.py anymore

4. Type 'python microcircuit.py' to start the simulation on a single process


---------------------------------------------------

Known issues:

- At least with PyNN 0.7.5 and NEST revision 10711, ConnectWithoutMultapses
  works correctly on a single process, but not with multiple MPI processes.

- When saving connections to file, ensure that pyNN does not create problems
  with single or nonexistent connections, for instance by adjusting
  lib/python2.6/site-packages/pyNN/nest/__init__.py from line 365 as follows:

  if numpy.size(lines) != 0:
      if numpy.shape(numpy.shape(lines))[0] == 1:
          lines = numpy.array([lines])
          lines[:,2] *= 0.001
          if compatible_output:
              lines[:,0] = self.pre.id_to_index(lines[:,0])
              lines[:,1] = self.post.id_to_index(lines[:,1])
      file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})

- To use saveConnections in parallel simulations, additionally ensure that
  pyNN does not cause a race condition where the directory is created by one
  process between the if statement and makedirs on another process: In
  lib/python2.6/site-packages/pyNN/recording/files.py for instance replace

  os.makedirs(dir)

  by 

  try: 
      os.makedirs(dir) 
  except OSError, e: 
      if e.errno != 17: 
          raise   
      pass

Reinstall pyNN after making these adjustments, so that they take effect
in your pyNN installation directory.
