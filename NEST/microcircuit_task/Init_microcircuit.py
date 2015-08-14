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

        # check whether we already have an instance type Init_microcircuit
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


if __name__ == "__main__":
    pass
