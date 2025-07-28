import importlib, pkg_resources, inspect
mod = importlib.import_module("inne")
print("inne version:", pkg_resources.get_distribution("inne").version)
print("has INNE :", hasattr(mod, "INNE"))
print("module file:", mod.__file__)



