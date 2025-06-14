import importlib, sys
lg = importlib.import_module("langgraph")
print("path:", lg.__file__)
print("add_subgraph ?", hasattr(lg.StateGraph, "add_subgraph"))
exit()
