# 导入所需的依赖包
import os

from h11 import ERROR
from sqlite_utils.cli import query

import config.load_env as env
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# 本章不使用transformer库，我们可以只关注transformer库中error级别的报警信息
import logging
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    MarkdownNodeParser,
    TokenTextSplitter
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, answer_correctness
from chatbot import rag

#设置日志级别
logging.basicConfig(level=logging.ERROR)
#加在API迷药
api_key = env.load_key()
api_base = env.load_url()

#配置llm
Settings.llm = OpenAILike(
    model = "qwen-plus",
    api_key = api_key,
    api_base = api_base,
    is_chat = True
)
#配置文本向量模型设置批处理大小和最大输入长度
Settings.embed_model = DashScopeEmbedding(
    mode_name = DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size = 6,
    max_input_length = 8192,
)

#定义问答函数
def ask(question,query_engine):
    # 更新提示模板
    rag.update_prompt_template(query_engine = query_engine)

    #  输出问题
    print('=' * 50) #使用乘法生成分割线
    print(f'问题:{question}')
    print('=' * 50 + '\n') #使用乘法生成分割线

    #获取答案
    response = query_engine.query(question)

    #输出答案
    print('回答：')
    if hasattr(response,'print_response_stream') and callable(response.print_response_stream):
        response.print_response_stream()
    else:
        print(response)

    #输出参考文档
    print('=' * 50)
    print('参考文档：')
    for i,source_node in enumerate(response.source_nodes,start = 1):
        print(f'文档{i}')
        print(source_node)
        print()
    print('=' * 50)

    return response
query_engine = rag.create_query_engine(rag.load_index())
response = ask('张伟是哪个部门的', query_engine)