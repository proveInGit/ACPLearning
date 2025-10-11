from chatbot import rag

# 第一次运行时启用（之后注释掉）
# rag.indexing(document_path="./docs", persist_path="./knowledge_base/test")

# 正常使用
index = rag.load_index(persist_path="./knowledge_base/test")
query_engine = rag.create_query_engine(index)
rag.ask("我们公司项目管理应该用什么工具？", query_engine)
