# RAG_node_1_prompting.py

"""PromptingNode — LLM 把自然语言 → JSON action list."""
import os, json
from string import Template
from openai import OpenAI                      # NVIDIA endpoint
from dotenv import load_dotenv

load_dotenv()
NGC_API_KEY = os.getenv("NGC_API_KEY")
if NGC_API_KEY is None:
    raise EnvironmentError("NGC_API_KEY not set")

class PromptingNode:
    def __init__(self) -> None:
        print("[LOG] PromptingNode initialized.")
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NGC_API_KEY,
        )
        # —— prompt 模板（只列可用函数/工具名称）——
        tool_list = [
            "select_rows", "sort_rows", "group_by_aggregate", "top_n",
            "filter_date_range", "add_derived_column", "rolling_average",
            "calculate_average", "calculate_mode", "calculate_median",
            "calculate_sum", "calculate_min", "calculate_max", "calculate_std",
            "calculate_variance", "calculate_percentile", "calculate_correlation",
            "calculate_covariance", "calculate_failure_rate", "count_rows",
            "calculate_delay_avg", "calculate_delay_avg_grouped",
            "graph_export", "plot_machine_avg_bar", "plot_concurrent_tasks_line"
        ]
        self.prompt_template = Template(r"""
        You are an assistant that MUST translate a user's natural‑language request
        into a raw JSON command describing a sequence of data‑processing steps.
        If the question restricts rows (e.g. “for Grinding jobs”), ALWAYS start with a select_rows action that applies that filter.

        JSON schema you MUST follow  ⬇︎
        {
          "actions": [
            {
              "function": "<one of: select_rows | sort_rows | calculate_average | calculate_mode | calculate_median>",
              "args": {
                // key–value pairs required by that function
              }
            },
            ...
          ]
        }
        

        ––––––––––––––––––––––––––––––––––––
        Columns available in the CSV
        ––––––––––––––––––––––––––––––––––––
        Job_ID · Machine_ID · Operation_Type · Material_Used · Processing_Time ·
        Energy_Consumption · Machine_Availability · Scheduled_Start ·
        Scheduled_End · Actual_Start · Actual_End · Job_Status · Optimization_Category

        ––––––––––––––––––––––––––––––––––––
        Example I/O
        ––––––––––––––––––––––––––––––––––––
        User input:
        "I need all jobs whose Optimization_Category is Low Efficiency AND
        Processing_Time ≤ 50, then sort them by Machine_Availability descending."

        Expected JSON output (RAW, no markdown):
        {
          "actions": [
            { "function": "select_rows",
              "args": { "column": "Optimization_Category", "condition": "== 'Low Efficiency'" }
            },
            { "function": "select_rows",
              "args": { "column": "Processing_Time", "condition": "<= 50" }
            },
            { "function": "sort_rows",
              "args": { "column": "Machine_Availability", "order": "desc" }
            }
          ]
        }

        ––––––––––––––––––––––––––––––––––––
        Reminder: if you need the scalar you just computed, insert {last_scalar} in your formula ,but ONLY if you actually calculated a scalar in a previous step.
        Now analyse the user's request below and output ONLY the JSON, no markdown, no explanations:
        \"\"\"$user_request\"\"\"
        """)

    def run(self, user_input: str) -> str:
        print("[LOG] PromptingNode running …")
        prompt = self.prompt_template.substitute(user_request=user_input)
        resp = self.client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=0.7,
            max_tokens=4096,
        )
        raw = resp.choices[0].message.content.strip()
        print(f"[LOG] LLM JSON:\n{raw}")
        # 快速 sanity check，确保 json 可解析
        _ = json.loads(raw)
        return raw
