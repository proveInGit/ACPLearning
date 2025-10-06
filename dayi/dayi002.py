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



# # temperature，top_p的默认值使用通义千问Max模型的默认值
# def print_qwen_stream_response(user_prompt, system_prompt, temperature=0.7, top_p=0.8, iterations=10):
#     for i in range(iterations):
#         print(f"输出 {i + 1} : ", end = "")
#         ## 防止限流，添加延迟
#         time.sleep(0.5)
#         response = get_qwen_stream_response(user_prompt, system_prompt, temperature, top_p)
#         output_content = ''
#         for chunk in response:
#             output_content += chunk
#         print(output_content)
#
#
# # 通义千问Max模型：temperature的取值范围是[0, 2)，默认值为0.7
# # 设置temperature=1.9
# print_qwen_stream_response(user_prompt = "马也可以叫做", system_prompt = "请帮我续写内容，字数要求是4个汉字以内。",
#                            temperature = 1.9)
user_question = "我是软件一组的，请问项目管理应该用什么工具"
knowledge = """公司项目管理工具有两种选择：
  1. **Jira**：对于软件开发团队来说，Jira 是一个非常强大的工具，支持敏捷开发方法，如Scrum和Kanban。它提供了丰富的功能，包括问题跟踪、时间跟踪等。

  2. **Microsoft Project**：对于大型企业或复杂项目，Microsoft Project 提供了详细的计划制定、资源分配和成本控制等功能。它更适合那些需要严格控制项目时间和成本的场景。

  在一般情况下请使用Microsoft Project，公司购买了完整的许可证。软件研发一组、三组和四组正在使用Jira，计划于2026年之前逐步切换至Microsoft Project。
"""
response = get_qwen_stream_response(
    user_prompt = user_question,
    # 将公司项目管理工具相关的知识作为背景信息传入系统提示词
    system_prompt = "你负责教育内容开发公司的答疑，你的名字叫公司小蜜，你要回答学员的问题。" + knowledge,
    temperature = 0.7,
    top_p = 0.8
)

for chunk in response:
    print(chunk, end = "")
