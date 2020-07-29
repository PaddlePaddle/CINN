from ..core_api.ir import *

def get_global_func(name):
    return Registry.get(name)

def register(name, override=False):
    def _register_fn(fn):
        Registry.register(name, override).set_body(PackedFunc(fn))
        return Registry.get(name)
    return _register_fn
