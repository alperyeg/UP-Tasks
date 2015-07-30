# ----------------------------------------------------------------------------
# This file will replace param_files network_params.py and sim_params.py
# All values will be stored in default_values and get accessed by getter_method
# . New values will be overwritten by applying setter_method.
# There are 3 versions:
# (1) run simulation only with default_values in config-file .yaml
# (2) run simulation with config-file .yaml can be changed by user. Specialized
# type will be needed ex. application/vnd.juelich.Simulation.NestConfig
# (3) run simulation with some options input from WUI by user
# ----------------------------------------------------------------------------
import yaml


class Init_microcircuit():
    class __Impl:
        """ Implementation of singleton interface """

        def __init__(self, arg):
            self.properties = arg

        def impl_id(self):
            return id(self)

    __instance = None

    def __init__(self, arg):

        # Check whether we already have an instance type Init_microcircuit
        if not Init_microcircuit.__instance:
            Init_microcircuit.__instance = Init_microcircuit.__Impl(arg)
        else:
            # __instance is None
            Init_microcircuit.__instance.properties = arg

        self.__dict__['_Init_mirocircuit__instance'] = Init_microcircuit.__instance

    def __getattr__(self, name):
        return getattr(self.__instance, name)

    def __setattr__(self, name, value):
        if "properties" in self.__dict__ and name in self.properties:
            self.properties[name] = value
        else:
            self.__dict__[name] = value

    # Numbers of neurons from which to record spikes
    def get_n_rec(self):
        N_full = self.properties['N_full']
        N_scaling = self.properties['params_dict']['nest']['N_scaling']
        frac_record_spikes = self.properties['params_dict']['nest']['frac_record_spikes']
        layers = self.properties['layers']
        pops = self.properties['pops']
        record_fraction = self.properties['params_dict']['nest']['record_fraction']
        n_record = self.properties['params_dict']['nest']['n_record']

        n_rec = {}
        for layer in layers:
            n_rec[layer] = {}
            for pop in pops:
                if record_fraction:
                    n_rec[layer][pop] = min(int(round(N_full[layer][pop] *
                                                      N_scaling *
                                                      frac_record_spikes)),
                                            int(round(N_full[layer][pop] *
                                                      N_scaling)))
                else:
                    n_rec[layer][pop] = min(n_record,
                                            int(round(N_full[layer][pop] *
                                                      N_scaling)))
        return n_rec




def init_config():
    import yaml
    with open('microcircuit.yaml', 'r') as f:
        conf = yaml.load(f)
    mc = Init_microcircuit(conf)
    return mc


if __name__ == "__main__":
    mc = init_config()

    with open('microcircuit.yaml', 'r') as f:
        conf2 = yaml.load(f)
    mc2 = Init_microcircuit(conf2)

    print "mc = ", mc.properties['params_dict']['nest']['K_scaling']
    mc.properties['params_dict']['nest']['K_scaling'] = 2
    print "mc2 = ", mc2.properties['params_dict']['nest']['K_scaling']

    print mc.get_n_rec()
