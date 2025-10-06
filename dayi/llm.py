import os
import time
import config.load_env as env
from openai import OpenAI

client = OpenAI(
    api_key = env.load_key(),
    base_url = env.load_url(),
)
def get_qwen_response(prompt):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            # system message 用于设置大模型的角色和任务
            {"role": "system", "content": ""},
            # user message 用于输入用户的问题
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content