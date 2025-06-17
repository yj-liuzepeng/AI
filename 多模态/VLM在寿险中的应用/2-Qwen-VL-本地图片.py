# -*- coding: utf-8 -*-
"""
本地图片识别 VLM 示例
依赖：
    pip install dashscope
    # 建议提前下载好图片，放在脚本同目录下
    # API Key 建议用环境变量管理：export DASHSCOPE_API_KEY=你的key
"""
import os
import dashscope
from dashscope import MultiModalConversation

# 从环境变量获取API Key
api_key = os.getenv('DASHSCOPE_API_KEY', '')
if not api_key:
    raise ValueError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置你的 DashScope API Key！")
dashscope.api_key = api_key

# 本地图片路径（需确保图片在脚本同目录下）
# local_file_path = 'file://1-Chinese-document-extraction.jpg'
# local_file_path = 'file://2-Japanese-document-extraction.jpg'
# local_file_path = 'file://3-French-document-extraction.jpg'
local_file_path = 'file://4-German-document-extraction.jpg'
# local_file_path = 'file://5-Korean-document-extraction.jpg'

# 构建多模态对话消息
messages = [
    {
        'role': 'system',
        'content': [
            {'text': 'You are a helpful assistant.'}
        ]
    },
    {
        'role': 'user',
        'content': [
            {'image': local_file_path},
            {'text': '图片里有什么东西?'}
        ]
    }
]

# 调用多模态模型
response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)

# 输出模型返回内容
print(response.output.choices[0].message.content[0]['text']) 