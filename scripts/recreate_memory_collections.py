"""三级记忆 collection 重建脚本：episodic / user_profile 换成中文 embedding

背景（为什么需要重建）：
  这两个 collection 最早是用 ChromaDB 英文默认模型（all-MiniLM-L6-v2，384 维）
  创建的，中文查询的向量召回近乎随机投影，情景记忆静默失效。
  换中文模型（bge-small-zh-v1.5，512 维）必须删库重建——embedding 配置在
  collection 创建时固化，改代码传新函数对已存在的 collection 无效。

  现在 memory/conversation_memory.py 创建时已传 make_embedding_function()，
  本脚本只负责把线上残留的旧 collection 换掉。注意：重复执行会清空并重建
  这两个 collection（丢数据），不是"重复执行无副作用"的幂等——只该跑一次。

损失说明：删除会清空开发期积累的情景记忆与用户画像（目前基本是冒烟测试
数据，损失≈0）。生产环境执行前务必先备份。

推荐执行顺序（app 运行中直接删，旧进程持有的句柄会短暂失效）：
  1. docker compose stop openmed
  2. docker compose run --rm openmed python scripts/recreate_memory_collections.py
  3. docker compose start openmed

用法：python scripts/recreate_memory_collections.py
"""
import sys
from pathlib import Path

# 让脚本能 import 项目根下的 core 包（standalone 运行时不在默认 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
from core.embedding import make_embedding_function  # noqa: E402

# 与 conversation_memory.py 中硬编码的名字保持一致
COLLECTIONS = ("episodic", "user_profile")


def load_env() -> dict:
    """解析 CHROMA_HOST/CHROMA_PORT，优先级：环境变量 > .env 文件 > 默认值。

    - 宿主机直跑：环境里没有这两个变量 → 读 .env → localhost:8001（映射容器 8000）
    - 容器内跑（docker compose run）：compose 注入 CHROMA_HOST=chromadb，
      必须优先于 .env 里的 localhost，否则连不上向量库服务
    """
    import os

    env = {"CHROMA_HOST": "localhost", "CHROMA_PORT": "8001"}
    for key in env:
        if os.environ.get(key):
            env[key] = os.environ[key]
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() in env and not os.environ.get(key.strip()):
                    env[key.strip()] = value.strip()
    return env


def main():
    env = load_env()
    host, port = env["CHROMA_HOST"], int(env["CHROMA_PORT"])

    # 只连服务端模式：重建是破坏性操作，连不上服务就停下，
    # 绝不在本地嵌入式副本上悄悄重建（那会把两条数据线搞混）
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    client.heartbeat()
    print(f"已连接 ChromaDB 服务: {host}:{port}")

    for name in COLLECTIONS:
        try:
            old = client.get_collection(name)
        except Exception:
            print(f"[{name}] 不存在，直接按中文 embedding 创建")
            old = None

        old_count = old.count() if old else 0
        if old:
            print(f"[{name}] 旧 collection 有 {old_count} 条数据（英文 embedding），将删除")
            client.delete_collection(name)

        # 重建：必须传中文 embedding 函数，否则又会落回英文默认模型
        col = client.get_or_create_collection(
            name,
            metadata={"description": f"OpenMed 三级记忆 {name}（中文 embedding）"},
            embedding_function=make_embedding_function(),
        )
        assert col.count() == 0, f"[{name}] 重建后 count 应为 0，实际 {col.count()}"
        print(f"[{name}] 已重建为空 collection（512 维中文向量），丢失旧数据 {old_count} 条")

    print("完成。下一步：启动 app 容器，跑 /chat 冒烟确认情景记忆检索无报错")


if __name__ == "__main__":
    main()
