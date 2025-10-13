# 导入所需的依赖包
import os
import dayi004
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
index = rag.load_index()
query_engine = index.as_query_engine(
    streaming=True,
    # 一次检索出 5 个文档切片，默认为 2
    similarity_top_k=5,
    llm=OpenAILike(
        model="qwen-plus",
        api_key=env.load_key(),
        api_base=env.load_url(),
        is_chat_model=True)
    )
# query_engine = rag.create_query_engine(rag.load_index())
response = ask('张伟是哪个部门的', query_engine)


# 定义评估函数
def evaluate_result(question, response, ground_truth):
    # 获取回答内容
    if hasattr(response, 'response_txt'):
        answer = response.response_txt
    else:
        answer = str(response)
    # 获取检索到的上下文
    context = [source_node.get_content() for source_node in response.source_nodes]

    # 构造评估数据集
    data_samples = {
        'question': [question],
        'answer': [answer],
        'ground_truth':[ground_truth],
        'contexts' : [context],
    }
    dataset = Dataset.from_dict(data_samples)

    # 使用Ragas进行评估
    score = evaluate(
        dataset = dataset,
        metrics=[answer_correctness, context_recall, context_precision],
        llm=Tongyi(model_name="qwen-plus"),
        embeddings=DashScopeEmbeddings(model="text-embedding-v3")
    )
    return score.to_pandas()

question = '张伟是哪个部门的'
ground_truth = '''公司有三名张伟，分别是：
- 教研部的张伟：职位是教研专员，邮箱 zhangwei@educompany.com。
- 课程开发部的张伟：职位是课程开发专员，邮箱 zhangwei01@educompany.com。
- IT部的张伟：职位是IT专员，邮箱 zhangwei036@educompany.com。
'''
evaluate_result(question = question, response = response, ground_truth = ground_truth)