## 作业2：阅读06-stock-bi-agent代码，回答如下问题：
### 1.  什么是前后端分离？

前后端分离是一种软件架构设计模式，将前端（负责渲染页面）和后端（负责处理数据、业务逻辑）解耦，通过HTTP API进行通信。

**特点：**
前端和后端可以并行开发；分工明确清晰；便于维护拓展

**工作流程：**
前端发送HTTP请求-》后端处理 -》 返给结果 -》前端渲染展示

### 2. 历史对话如何存储，以及如何将历史对话作为大模型的下一次输入？

采用 双数据库架构 实现历史对话的存储与复用：

业务数据库（ sever.db ）通过 ChatSessionTable 和 ChatMessageTable 两张表持久化存储对话元数据和完整消息记录（包括 role、content、时间戳、反馈等），支持查询、展示和用户管理；
Agent 专用数据库（ conversations.db ）通过 OpenAI Agents SDK 的 AdvancedSQLiteSession 自动管理，以 session_id 为桥梁关联两个数据库。

每次对话时，系统先用 session_id 检查或创建会话，将用户消息存入 sever.db ，然后 AdvancedSQLiteSession 自动从 conversations.db 读取该会话的所有历史消息（system + 历轮 user/assistant），构建完整上下文发送给大模型，大模型基于完整历史生成回答后，再流式返回给用户并同时保存到两个数据库中，从而实现多轮对话的上下文连贯性。
