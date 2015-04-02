#! /usr/bin/env python

import nest
import nest.voltage_trace
import pylab
from active_worker.task import task


@task
def neuron_noise_task():
    '''
        Task Manifest Version: 1
        Full Name: neuron_noise_task
        Caption: One neuron with noise
        Author: nest-initiative.org
        Description: |
            A minimal model with just one neuron and Poisson noise input.
        Categories:
            - NEST
        Compatible_queues: ['cscs_viz', 'epfl_viz']
        Accepts:

        Returns:
            res: image/png

    '''
    nest.ResetKernel()
    nest.SetKernelStatus({"overwrite_files": True})

    neuron = nest.Create("iaf_neuron")
    noise = nest.Create("poisson_generator", 2)
    voltmeter = nest.Create("voltmeter")

    nest.SetStatus(noise, [{"rate": 80000.0}, {"rate": 15000.0}])

    # get default_location of 'to_file' and turn it into manual location
    nest.SetStatus(voltmeter, [{"to_memory": True, "withtime": True}])
    # nest.SetStatus(voltmeter, [{"filenames": newname}])

    nest.ConvergentConnect(noise, neuron, [1.2, -1.], [1.0, 1.0])
    nest.Connect(voltmeter, neuron)
    nest.Simulate(500.0)

    # print nest.GetStatus(voltmeter, "file_extension")[0]
    # print type(nest.GetStatus(voltmeter, "filenames")[0])
    # print nest.GetStatus(voltmeter, "global_id")[0]
    # print nest.GetStatus(voltmeter, "local_id")[0]
    # print nest.GetStatus(voltmeter, "to_screen")[0]
    # print nest.GetStatus(voltmeter, "to_memory")[0]
    # print nest.GetStatus(voltmeter, "to_file")[0]
    # print nest.sli_pop()

    # TODO: check the existence of exported file & move it to somewhere else

    # print nest.GetStatus(voltmeter)

    # ... Visualization : Show image
    nest.voltage_trace.from_device(voltmeter)
    # print plotids
    # nest.voltage_trace.show()
    # name = __filename__+".png"
    # nest.voltage_trace.show()
    # nest.voltage_trace.savefig('foo.png')

    # TODO: write result as image to file, see voltage_trace.py
    out_file_name = 'neuron_noise_task.png'
    with open(out_file_name, 'w') as report_path:
        pylab.savefig(report_path, dpi=100)

    return neuron_noise_task.task.uri.save_file(mime_type='image/png',
                                                src_path=out_file_name,
                                                dst_path=out_file_name)


if __name__ == '__main__':
    neuron_noise_task()
