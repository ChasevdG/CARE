PE_REGISTRY = {}

def register_PE(name, cls):
    PE_REGISTRY[name] = cls