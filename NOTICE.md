# 来源与说明

- 本项目以公开 RAG 入门学习资料中的「入库 → 检索 → 生成」链路为学习起点，重新组织为本地知识库问答应用。不包含课程视频、课件或其他第三方原始素材。
- 代码、测试、实验脚本和文档在 AI 助手协助下完成。
- 示例资料、检索题集和固定输入全部为自编内容，不是真实企业制度、产品政策或用户反馈。
- 实验调用阿里云百炼（DashScope）的 qwen-plus 和 text-embedding-v4，运行会产生费用，由运行者自行承担。`results/retrieval/embedding_cache.json` 是对自编文本调用 text-embedding-v4 得到的向量。
- 依赖见 `requirements.txt`，联网实验另需 openai，画图另需 matplotlib。第三方库的版权归各自作者。
