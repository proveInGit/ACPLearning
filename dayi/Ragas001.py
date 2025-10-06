import multiprocessing
import os
os.environ["USE_THREAD_POOL"] = "True"

multiprocessing.set_start_method('spawn', force=True)

from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness
from config.load_env import load_key

if __name__ == '__main__':
    # 加载api_key
    api_key = load_key().strip()
    #初始化llm和Embeddings
    llm = Tongyi(
        model_name = "qwen-turbo",
        api_key = api_key
    )
    embeddings = DashScopeEmbeddings(
        model = "text-embedding-v3",
        dashscope_api_key = api_key
    )
    #构造测试数据
    data_sample = {
        'question' : ["张伟是哪个部门的"] * 3,
        'answer' : [
            "根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。",
            "张伟是人事部的",
            "张伟是教研部的"
         ],
        'ground_truth' : ["张伟是教研部的"] * 3
    }
    dataset = Dataset.from_dict(data_sample)
    #开始评估
    print("现在开始评估...请等待")
    score = evaluate(
        dataset = dataset,
        metrics = [answer_correctness],
        llm = llm,
        embeddings = embeddings
    )
    #输出结果
    print(score.to_pandas())
