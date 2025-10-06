from openai import OpenAI
import os

def reasoning_model_response(user_prompt, system_prompt="你是一个编程助手。", model = "qwen3-235b-a22b-thinking-2507"):

    # 初始化客户端
    client = OpenAI(
        api_key =
    )