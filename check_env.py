reqs = [
    "langgraph", "langchain", "langchain_openai", "openai",
    "pandas", "networkx", "pyvis", "neo4j", "matplotlib", "plotly"
]
missing = []
for r in reqs:
    try:
        __import__(r.replace("-", "_"))
    except ImportError:
        missing.append(r)
if missing:
    print("缺少包：", missing)
else:
    print("🎉 解释器与依赖全部就绪！")
