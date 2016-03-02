#!/usr/bin/env python

# =============================================================================
# Initialization
# =============================================================================

from active_worker.task import task
from task_types import TaskTypes as tt
import unicore_client
import os


def load_local_file(name):
    """ loads local file and returns contents as string """
    file_path = os.path.join(os.path.dirname(__file__), name)
    with open(file_path) as f:
        return f.read()


@task
def cch_vista_submit_task(inputdata_spinnaker, inputdata_nest, run_script,
                          aux_script):
    '''
        Task Manifest Version: 1
        Full Name: cch_vista_submit_task
        Caption: cch_vista_submit_task
        Author: Elephant-Developers
        Description: |
            This task submitts a script to an HPC, calculates the pairwise 
            correlation and stores an p-value significance matrix, which can be 
            read in by the `vista` visualization framework.
        Categories:
            - FDAT
        Compatible_queues: ['cscs_viz']
        Accepts:
            inputdata_spinnaker:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file generated from spinnaker simulation.
            inputdata_spinnaker:
                type: application/unknown
                description: Input file that contains spiking data from a
                    HDF5 file from nest simulation.
            script:
                type: application/unkown
                description: Script which will be executed on an HPC.

        Returns:
            res: application/unknown
    '''
    # TODO add the h5py wrapper script
    spinnaker_data_path = cch_vista_submit_task.task.uri.get_file(inputdata_spinnaker)
    nest_data_path = cch_vista_submit_task.task.uri.get_file(inputdata_nest)
    submit_script_path = cch_vista_submit_task.task.uri.get_file(run_script)
    collect_script_path = cch_vista_submit_task.task.uri.get_file(aux_script)
    code = {'To': '{}'.format(os.path.split(submit_script_path)[1]), 'Data': load_local_file('{}'.format(submit_script_path))}
    collect_script = {'To': '{}'.format(os.path.split(collect_script_path))[1], 'Data': load_local_file('{}'.format(collect_script_path))}
    spinnaker_data = {'To': '{}'.format(os.path.split(spinnaker_data_path))[1], 'Data': load_local_file('{}'.format(spinnaker_data_path))}
    nest_data = {'To': '{}'.format(os.path.split(nest_data_path))[1], 'Data': load_local_file('{}'.format(nest_data_path))}
    inputs = [code, collect_script, spinnaker_data, nest_data]

    oauth_token = cch_vista_submit_task.task.uri.get_oauth_token()
    auth = unicore_client.get_oidc_auth(oauth_token)
    job = {}
    job['ApplicationName'] = 'Elephant'
    job['Parameters'] = {'INPUT': submit_script_path}
    job['Environment'] = {'spinnaker_data': os.path.split(spinnaker_data_path)[1],
                          'nest_data': os.path.split(nest_data_path)[1]}
    job['User postcommand'] = 'python {0} {1}'.format(os.path.split(collect_script_path)[1], spinnaker_data)
    job['RunUserPostCommandOnLoginNode'] = 'false'

    base_url = unicore_client.get_sites()['JURECA']['url']
    job_url = unicore_client.submit(os.path.join(base_url, 'jobs'), job, auth,
                                    inputs)
    print "Submitting to {}".format(job_url)
    unicore_client.wait_for_completion(job_url, auth)


if __name__ == '__main__':
    inputdata_spinnaker = tt.URI('application/unknown', 'spikes_L5E.h5')
    inputdata_nest = tt.URI('application/unknown', 'spikes_L5E.h5')
    script = ''
    cch_vista_submit_task(inputdata_spinnaker, inputdata_nest, script)
