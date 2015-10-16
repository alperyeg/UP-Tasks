import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest  # import NEST module

from active_worker.task import task


@task
def single_neuron_task(spike_times, sim_duration):
    '''
       Task Manifest Version: 1
       Full Name: single_neuron_task
       Caption: Single neuron
       Author: NEST Developers
       Description: |
           This script simulates a neuron stimulated by spikes
           with predefined times and creates a plot of the membrane
           potential trace.
           Simulator: NEST (http://nest-simulator.org)
       Categories:
           - NEST
       Compatible_queues: ['cscs_viz', 'cscs_bgq', 'epfl_viz']
       Accepts:
           spike_times:
               type: list(double)
               description: Spike times in ms at which the neuron is stimulated (e.g., [10, 50]).
           sim_duration:
               type: double
               description: Simulation duration in ms (e.g., 100).
       Returns:
           res: image/png
    '''

    nest.set_verbosity('M_WARNING')  # reduce NEST output
    nest.ResetKernel()  # reset simulation kernel
    # create LIF neuron with exponential synaptic currents
    neuron = nest.Create('iaf_psc_exp')
    # create a voltmeter
    voltmeter = nest.Create('voltmeter', params={'interval': 0.1})
    # create a spike generator
    spikegenerator = nest.Create('spike_generator')
    # ... and let it spike at predefined times
    nest.SetStatus(spikegenerator, {'spike_times': spike_times})
    # connect spike generator and voltmeter to the neuron
    nest.Connect(spikegenerator, neuron)
    nest.Connect(voltmeter, neuron)
    # run simulation for sim_duration
    nest.Simulate(sim_duration)
    # read out recording time and voltage from voltmeter
    times = nest.GetStatus(voltmeter)[0]['events']['times']
    voltage = nest.GetStatus(voltmeter)[0]['events']['V_m']
    # plot results
    plt.plot(times, voltage)
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    filename = 'single_neuron_task.png'
    plt.savefig(filename, dpi=300)

    return single_neuron_task.task.uri.save_file(mime_type='image/png',
                                                 src_path=filename,
                                                 dst_path=filename)

if __name__ == '__main__':
    spike_times = [10., 50.]
    sim_duration = 100.
    single_neuron_task(spike_times, sim_duration)
