import config.load_env as env
import os
from openai import OpenAI
client = OpenAI(
    api_key = env.load_key(),
    base_url = env.load_url()
)

import time


def get_qwen_stream_response(user_prompt, system_prompt, temperature, top_p):
    response = client.chat.completions.create(
        model = "qwen-max",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature = temperature,
        top_p = top_p,
        stream = True
    )

    for chunk in response:
        yield chunk.choices[0].delta.content


# temperature，top_p的默认值使用通义千问Max模型的默认值
def print_qwen_stream_response(user_prompt, system_prompt, temperature=0.7, top_p=0.8, iterations=10):
    for i in range(iterations):
        print(f"输出 {i + 1} : ", end = "")
        ## 防止限流，添加延迟
        time.sleep(0.5)
        response = get_qwen_stream_response(user_prompt, system_prompt, temperature, top_p)
        output_content = ''
        for chunk in response:
            output_content += chunk
        print(output_content)


# 通义千问Max模型：temperature的取值范围是[0, 2)，默认值为0.7
# 设置temperature=1.9
print_qwen_stream_response(user_prompt = "马也可以叫做", system_prompt = "请帮我续写内容，字数要求是4个汉字以内。",
                           temperature = 1.9)

