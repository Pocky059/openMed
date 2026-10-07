"""pytest 全局配置。

强制 HuggingFace 离线：单测不允许联网下载模型（CLAUDE.md 约定「56 个单测，
不联网」）。意图识别语义路（core/embedding.py 的 get_local_embedder）在测试
环境会因缓存缺失快速失败，自动退字符 n-gram 哈希兜底——这正是我们要测的
降级行为，且保证测试速度与确定性。

注意：必须在本文件 import 任何会触发模型加载的模块之前设置环境变量，
pytest 加载 conftest 早于测试模块，时序上是安全的。
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
