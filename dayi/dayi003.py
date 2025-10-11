from sqlite_utils.cli import query
from tomlkit import document

import config.load_env as env
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from chatbot import rag
import logging
logging.basicConfig(level = logging.ERROR)


# load documents
print('=' * 50)
print('正在解析文件...')
print('=' * 50 + '\n')
#LlmaIndex提供了SimpleDirectoryReader方法，可以直接将制定文件夹中的文件加载为document对象，对应着解析过程
documents = SimpleDirectoryReader('./docs').load_data()

print('=' * 50)
print('创建索引...')
print('=' * 50 + '\n')

# from_documents方法包含切片与建立索引的步骤
index  = VectorStoreIndex.from_documents(
    documents,
    #指定embeddings模型
    embed_model = DashScopeEmbedding(
        model_name = DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    )
)
print('=' * 50)
print('创建提问索引...')
print('=' * 50 + '\n')

query_engine = index.as_query_engine(
    streaming = True,
    llm = OpenAILike(
        model = "qwen-plus",
        api_key = env.load_key(),
        api_base = env.load_url(),
        is_chat_model = True
    )
    )
print('=' * 50)
print('正在生成回复...')
print('=' * 50 + '\n')
streaming_response = query_engine.query("我们公司项目管理应该用什么工具")
print("回答是：")
#采用流式输出
streaming_response.print_response_stream()

