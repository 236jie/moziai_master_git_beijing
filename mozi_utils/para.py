import inspect

def my_function(a, b=2, *args, c: int, **kwargs):
    pass

signature = inspect.signature(my_function)
for name, param in signature.parameters.items():
    print(f"Parameter: {name}, Type: {param.annotation}, Default: {param.default}")