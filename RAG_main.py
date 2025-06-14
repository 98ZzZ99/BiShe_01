# RAG_main.py

"""Entry point — builds and runs the LangGraph StateGraph pipeline."""
from dotenv import load_dotenv, find_dotenv
from RAG_graph_config import build_graph
from RAG_subgraph_tabular_react import build_tabular_react_subgraph

# — 在任何其他 import 之前加载 .env —
load_dotenv(find_dotenv(), override=True, verbose=True) # find_dotenv()	递归向上寻找最近的 .env 文件并返回完整路径。override 允许 .env 中的键值覆盖进程里已有的同名环境变量 （默认是不覆盖的）。 verbose=True 允许打印加载的 .env 文件路径

def main() -> None:
    print("[LOG] Program started (LangGraph version).")
    graph = build_graph()   # 得到 CompiledStateGraph
    user_input = input("Please enter your request: ")
    result_state = graph.invoke({"user_input": user_input}) # 接收一份 初始状态字典，键名必须符合 PipelineState 里定义的字段名
    # ExecutionNode 会把结果对象放到 execution_output。执行完后返回的同样是一份字典，包含执行过程中被不断填充的 processed_input / llm_json / execution_output
    print("\n[PIPELINE OUTPUT]\n", result_state.get("execution_output"))
    print("[LOG] Program execution finished.")

if __name__ == "__main__":  # __name__ 是 Python 每个模块启动时自动注入的内置变量。如果这个脚本被别的文件 import，则 __name__ 就等于模块实际文件名（去掉扩展名）
    main()


