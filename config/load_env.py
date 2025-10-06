from dotenv import load_dotenv
import os
#在模块加载的时候就加载环境变量
load_dotenv()

def load_key():
    #读取API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    return api_key

def load_url():
    #读取API URL
    base_url = os.getenv("BASE_URL")
    return base_url
