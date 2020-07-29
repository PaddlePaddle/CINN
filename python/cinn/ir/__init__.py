from ..core_api.ir import *

def get_global_func(name):
    return Registry.get(name)

def register(name, override=False):
    def _register_fn(fn):
        Registry.register(name, override).set_body(PackedFunc(fn))
        return Registry.get(name)
    return _register_fn

def register_packed_func(name, override=False):
    def _register(fn):
        def _packed(args, rv):
            _args = []
            for i in range(len(args)):
                _args.append(args[i])
            r = fn(*_args)
            rv.set(r)
        Registry.register(name, override).set_body(PackedFunc(_packed))
        return Registry.get(name)
    return _register
