from ..core_api.ir import *

def get_global_func(name):
    return Registry.get(name)
