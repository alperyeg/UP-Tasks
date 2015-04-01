import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import os.path
from random import randint
from active_worker.task import task
from task_types import TaskTypes as tt

# This decoration is deprecated, from OLD-VERSION of task-sdk

_task_full_name = "elephant_cv_task"
_task_caption = "execute cv-function task"
_task_author = "Long Phan"
_task_description = "Input data is randomized inside function, used for\
    calculate cv and output an image/png as result"
_task_categories = ['elephant']
_task_compatible_queues = ['cscs_viz']


@task(returns=tt.URIType('image/png'))
def elephant_cv_task():

    # create data numpy for cv
    data_list = []
    for i in xrange(98):
        range_size = randint(1, 10)
        data_list.append(np.random.uniform(0, 1000, range_size))
    # print data_list
    # print len(data_list)
    # res: list of results calculated by cv-function
    res = []
    # pass data as input-parameter to cv-function
    for i in data_list:
        res.append(cv(i))

    # stdout of result
    print res

    export_hdf5(data_list, res)
    plt = export_matplotlib(res)

    # export visualized result as histogram-image at U.P.
    out_file_name = 'report_cv.png'
    with open(out_file_name, 'w') as report_path:
        plt.savefig(report_path, dpi=100)
    # blue_config_path = elephant_cv_task.task.uri.get_file(blueconfig_uri)
    dst_name = 'result_cv.png'
    return elephant_cv_task.task.uri.save_file(mime_type='image/png',
                                               src_path=out_file_name,
                                               dst_path=dst_name)


def export_hdf5(data_list, res):
    file = h5py.File("hdf5_cv_up.h5", 'w')
    dataset = file.create_dataset("dset", (len(data_list), ),
                                  h5py.h5t.NATIVE_DOUBLE)
    dataset[...] = res
    file.close()


def export_matplotlib(data):
    plt.xlabel('CV_data')
    plt.ylabel('Histogram')
    plt.title('Histogram of CV')
    plt.hist(data, bins=10)
    width = 1000
    plt.axis([0, width, 0, 100])
    plt.grid(True)
    # plt.show()
    return plt


def cv(data_inp):
    # print len(isis)
    isis = np.array([])

    numpy_data = [data_inp]

    # print ".... len(numpy_data)", len(numpy_data)
    for st in numpy_data:
        # print ".... st = ", st
        # print ".... type(st) =", type(st)
        # print ".... len(st) =", len(st)
        if len(st) > 1:
            isis = np.hstack([isis, np.diff(st)])

    if len(isis) == 0:
        CV = 0.
    #    print "CV is 0."
    else:
        CV = isis.std() / isis.mean()
    #    print "len(isis) = ", len(isis)
    #    print "isis.std() = ", isis.std()
    #    print "isis.mean() = ", isis.mean()
    #    print "CV = ", CV

    return CV

if __name__ == '__main__':
    # import sys
    # assert(len(sys.argv) == 2)
    # uri = tt.URI('application/vnd.bbp.Simulation.BlueConfig', sys.argv[1])
    # elephant_cv_task(uri)
    elephant_cv_task()
