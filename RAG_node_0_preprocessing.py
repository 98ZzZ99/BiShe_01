# RAG_node_0_preprocessing.py

class PreprocessingNode:
    def __init__(self) -> None:
        print("[LOG] PreprocessingNode initialized.")

    def run(self, user_input: str) -> str:
        print("[LOG] PreprocessingNode running …")
        out = user_input.strip()
        print(f"[LOG] Preprocessing completed. Output: {out}")
        return out


