# Milvus + LangChain 中文终端问答 Demo

一个基于 **Milvus Lite + LangChain + DeepSeek** 的中文终端问答项目。  
项目通过向量检索实现“用户资料知识库问答”，并结合意图识别支持统计、详情查询、随机抽样与通用闲聊。

---

## 1. 项目亮点

- 中文终端交互：支持连续多轮对话，适合本地演示与教学。
- 本地向量库：使用 Milvus Lite（单文件 DB）构建和查询向量索引。
- RAG 问答：基于相似度检索结果进行资料问答。
- 混合意图路由：规则识别 + LLM 精修，提升意图判断准确性。
- 多种查询能力：总量统计、姓氏统计、人物详情、随机资料、知识库问答、通用问答。
- 离线兜底向量化：当 HuggingFace Embedding 不可用时可回退本地哈希向量。
- 对话记忆：保留近期对话并自动生成摘要，提升多轮一致性。
- 可视化终端体验：Rich 输出、进度条、思考状态提示、输入输出颜色区分。

---

## 2. 功能实现说明

### 2.1 数据与索引

- 启动时自动生成指定数量的中文用户画像数据（默认 500 条）。
- 将画像文本化后进行向量化，并写入 Milvus Lite 集合。
- 如果已存在同规模索引则直接复用，避免重复重建。

### 2.2 意图识别与路由

- 先进行规则意图识别（正则 + 关键词），快速处理确定性请求。
- 对 `GENERAL_CHAT / KB_QA` 这类模糊请求，再用 LLM 二次精修意图。
- 针对不同意图走不同处理链路：
  - `TOTAL_COUNT`：统计知识库总条数
  - `SURNAME_COUNT`：统计指定姓氏人数
  - `PERSON_DETAIL`：按姓名检索详情（支持同名候选选择）
  - `RANDOM_PROFILE`：随机抽样用户资料
  - `KB_QA`：检索增强问答
  - `GENERAL_CHAT`：通用聊天

### 2.3 RAG 问答链路

- 按意图决定是否检索向量知识库。
- 将系统提示词、历史对话摘要、检索资料、当前问题拼装为 Prompt。
- 使用流式输出逐 token 在终端展示回复。
- 每轮结束后写入对话历史并更新摘要。

### 2.4 终端交互体验

- 支持中文输入法与行内编辑（`prompt-toolkit`）。
- 用户输入与“小咪”输出采用不同颜色区分。
- 执行耗时任务时显示“思考中”状态，减少空窗感。
- 索引构建过程带进度条，退出时自动落盘会话记录。

---

## 3. 技术栈与框架

### 3.1 语言与运行环境

- Python 3.10+（建议 3.11/3.12+）

### 3.2 主要依赖

- `langchain` / `langchain-community`
- `langchain-openai`
- `langchain-huggingface`
- `langchain-milvus`
- `pymilvus[model,milvus_lite]`
- `sentence-transformers`
- `faker`
- `rich`
- `prompt-toolkit`
- `python-dotenv`

---

## 4. 快速开始

## 4.1 克隆并进入项目

```bash
git clone <your-repo-url>
cd 426_milvusTry
```

## 4.2 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4.3 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少配置：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## 4.4 运行项目

```bash
python main.py
```

首次运行会自动：

1. 生成 mock 用户资料；
2. 构建 Milvus 向量索引；
3. 进入终端对话模式。

---

## 5. 环境变量与配置项

### 5.1 `.env` 关键变量

- `DEEPSEEK_API_KEY`：必填，DeepSeek API Key。
- `EMBEDDING_BACKEND`：向量化后端策略。
  - `auto`：优先 HuggingFace，失败回退 `local_hash`
  - `hf`：仅使用 HuggingFace，失败时报错
  - `local_hash`：仅使用本地哈希向量（离线兜底）
- `HF_ENDPOINT`：可选，HuggingFace 镜像地址（默认 `https://hf-mirror.com`）。
- `HF_HUB_DOWNLOAD_TIMEOUT`：可选，模型下载超时时间（秒）。

### 5.2 代码内默认配置（`config.py`）

- 模型：`deepseek-chat`
- 向量模型：`BAAI/bge-small-zh-v1.5`
- 向量库路径：`./milvus_data/demo.db`
- 集合名：`people_profiles`
- 检索 TopK：`5`
- 默认数据量：`500`
- 批次写入：`100`
- 历史记忆轮数：`10`
- 请求超时：`60s`
- 最大重试：`3`

---

## 6. 目录结构

```text
426_milvusTry/
├── main.py               # 程序入口与主对话流程
├── config.py             # 配置加载与系统提示词
├── data_generator.py     # 中文 mock 画像生成与文本化
├── intent_router.py      # 规则意图识别与解析
├── rag_chain.py          # RAG 问答引擎（流式输出+记忆摘要）
├── terminal_ui.py        # Rich + prompt_toolkit 终端 UI
├── vector_store.py       # Milvus Lite 存储与检索操作
├── requirements.txt      # 依赖列表
├── .env.example          # 环境变量样例
├── dataset/              # 生成的数据集 JSON
├── milvus_data/          # Milvus Lite 数据文件
└── chat_history/         # 会话历史输出目录
```

---

## 7. 使用示例

可尝试以下问题：

- `知识库有多少条数据？`
- `数据库里有多少条数据中的姓名是姓李的？`
- `张三的具体信息`
- `随便给我一个用户资料`
- `北京从事后端工程师的人多吗？`

---

## 8. 运行机制（高层流程）

1. 启动加载配置与依赖；
2. 检查/构建向量索引；
3. 读取用户输入；
4. 意图识别（规则 + LLM 精修）；
5. 根据意图执行统计/查询/RAG；
6. 流式返回答案并展示引用资料；
7. 更新对话记忆与摘要；
8. 退出时持久化历史会话到 `chat_history/`。

---

## 9. 常见问题（FAQ）

### Q1：启动时报 `缺少 DEEPSEEK_API_KEY`？

请在 `.env` 中配置 `DEEPSEEK_API_KEY`，或通过系统环境变量导出后再运行。

### Q2：HuggingFace 模型下载慢或失败？

- 将 `EMBEDDING_BACKEND=auto`（推荐）；
- 配置 `HF_ENDPOINT=https://hf-mirror.com`；
- 或直接使用 `EMBEDDING_BACKEND=local_hash` 走离线兜底。

### Q3：Milvus 报数据库占用？

确保没有其他进程同时占用同一 `milvus_data/demo.db`，关闭旧终端后重试。

### Q4：如何重建索引？

删除 `milvus_data/` 后重新运行，或修改配置中的数据规模触发重建。

---

## 10. 可扩展方向

- 接入真实业务数据（CSV/DB/API）替代 mock 数据；
- 添加更细粒度的意图分类与工具调用；
- 引入 rerank 与混合检索提升召回质量；
- 对接 Web 前端与会话管理；
- 增加自动化测试、评估集与基准脚本。

---

## 11. 许可证

如需开源发布，请在仓库补充 `LICENSE` 文件（例如 MIT）。

