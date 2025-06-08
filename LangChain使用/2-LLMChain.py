#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.agents import AgentType

# 你需要在环境变量中添加 OPENAI_API_KEY 和 SERPAPI_API_KEY
#os.environ["OPENAI_API_KEY"] = '*******'
os.environ["SERPAPI_API_KEY"] = 'becfe1cd81ac25acff4922247ffd0092819df56a6ebeb415f6e7e28f66dfee79'
 
# 设置通义千问API密钥
os.environ["DASHSCOPE_API_KEY"] = 'sk-f50e8206227745b4925d04e4d7dcc744'

# 加载模型
llm = Tongyi(model_name="qwen-turbo")  # 使用通义千问qwen-turbo模型

# 加载 serpapi 工具
tools = load_tools(["serpapi"])
 
"""
agent：代理类型  
    zero-shot-react-description: 根据工具的描述和请求内容的来决定使用哪个工具（最常用）
    react-docstore: 使用 ReAct 框架和 docstore 交互, 使用Search 和Lookup 工具, 前者用来搜, 后者寻找term, 举例: Wipipedia 工具
    self-ask-with-search 此代理只使用一个工具: Intermediate Answer, 它会为问题寻找事实答案(指的非 gpt 生成的答案, 而是在网络中,文本中已存在的), 如 Google search API 工具
    conversational-react-description: 为会话设置而设计的代理, 它的prompt会被设计的具有会话性, 且还是会使用 ReAct 框架来决定使用来个工具, 并且将过往的会话交互存入内存
"""
# 工具加载后需要初始化，verbose=True 代表打印执行详情
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 
# 运行 agent
agent.run("今天是几月几号?历史上的今天有哪些名人出生")


# In[1]:


# os.getenv("SERPAPI_API_KEY")

