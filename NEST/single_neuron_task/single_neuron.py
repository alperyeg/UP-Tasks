import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest  # import NEST module


def single_neuron(spike_times, sim_duration):
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
    filename = 'single_neuron.png'
    plt.savefig(filename, dpi=300)

if __name__ == '__main__':
    spike_times = [10., 30., 70.]
    sim_duration = 100.
    single_neuron(spike_times, sim_duration)