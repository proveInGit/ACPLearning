from dashscope.common.env import api_key
from langchain_community.llms.tongyi import Tongyi
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall,context_precision
import os
from config.load_env import load_key,load_url

data_samples = {
    'question': [
        '张伟是哪个部门的？',
        '张伟是哪个部门的？',
        '张伟是哪个部门的？'
    ],
    'answer': [
        '根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。',
        '张伟是人事部门的',
        '张伟是教研部的'
    ],
    'ground_truth':[
        '张伟是教研部的成员',
        '张伟是教研部的成员',
        '张伟是教研部的成员'
    ],
    'contexts' : [
        ['提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ', '绩效管理部 韩杉 李⻜ I902 041 ⼈⼒资源'],
        ['李凯 教研部主任 ', '牛顿发现了万有引力'],
        ['牛顿发现了万有引力', '张伟 教研部工程师，他最近在负责课程研发'],
    ],
}
dataset = Dataset.from_dict(data_samples)
score = evaluate(
    dataset = dataset,
    metrics = [context_precision,context_recall],
    llm = Tongyi(model_name = "qwen-plus", api_key = load_key(), api_url = load_url())
)
score.to_pandas()