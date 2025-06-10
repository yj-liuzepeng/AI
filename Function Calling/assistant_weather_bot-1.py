import os
import requests
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool

# 定义资源文件根目录
ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒


functions_desc = [
    {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气情况",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "城市编码，例如：110000（北京）",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_news",
        "description": "获取今日热点新闻",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "新闻类别，例如：top（头条）、society（社会）、tech（科技）等",
                },
                "count": {
                    "type": "integer",
                    "description": "获取新闻条数，默认10条",
                }
            },
            "required": ["category"],
        },
    }
]

# ====== 天气查询工具实现 ======
@register_tool('get_current_weather')
class WeatherTool(BaseTool):
    """
    天气查询工具，通过高德地图API查询指定位置的天气情况。
    """
    description = '获取指定位置的当前天气情况'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市名称，例如：北京',
        'required': True
    }, {
        'name': 'adcode',
        'type': 'string',
        'description': '城市编码，例如：110000（北京）',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        location = args['location']
        adcode = args.get('adcode', None)
        
        return self.get_weather_from_gaode(location, adcode)
    
    def get_weather_from_gaode(self, location: str, adcode: str = None) -> str:
        """调用高德地图API查询天气"""
        gaode_api_key = "xxxx"  # 高德API Key
        base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        
        params = {
            "key": gaode_api_key,
            "city": adcode if adcode else location,
            "extensions": "base",  # 可改为 "all" 获取预报
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == '1' and data.get('lives'):
                    weather_info = data['lives'][0]
                    result = f"天气查询结果：\n城市：{weather_info.get('city')}\n天气：{weather_info.get('weather')}\n温度：{weather_info.get('temperature')}°C\n风向：{weather_info.get('winddirection')}\n风力：{weather_info.get('windpower')}\n湿度：{weather_info.get('humidity')}%\n发布时间：{weather_info.get('reporttime')}"
                    return result
                else:
                    return f"获取天气信息失败：{data.get('info', '未知错误')}"
            else:
                return f"请求失败：HTTP状态码 {response.status_code}"
        except Exception as e:
            return f"获取天气信息出错：{str(e)}"

# ====== 新闻查询工具实现 ======
@register_tool('get_news')
class NewsTool(BaseTool):
    """
    新闻查询工具，通过聚合数据API获取热点新闻。
    """
    description = '获取今日热点新闻'
    parameters = [{
        'name': 'category',
        'type': 'string',
        'description': '新闻类别，例如：top（头条）、society（社会）、tech（科技）等',
        'required': True
    }, {
        'name': 'count',
        'type': 'integer',
        'description': '获取新闻条数，默认10条',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        category = args['category']
        count = args.get('count', 10)
        
        return self.get_news_from_juhe(category, count)
    
    def get_news_from_juhe(self, category: str, count: int = 10) -> str:
        """调用聚合数据API获取热点新闻"""
        # 聚合数据API配置
        juhe_api_key = "xxx"  # 需要替换为你的聚合数据API Key
        base_url = "http://v.juhe.cn/toutiao/index"
        
        # 新闻类别映射
        category_map = {
            'top': 'top',
            'society': 'shehui',
            'tech': 'keji',
            'sports': 'tiyu',
            'entertainment': 'yule',
            'finance': 'caijing',
            'military': 'junshi',
            'education': 'edu',
            'car': 'auto'
        }
        
        # 获取对应的新闻类别
        news_category = category_map.get(category.lower(), 'top')
        
        params = {
            'key': juhe_api_key,
            'type': news_category
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('error_code') == 0:
                    news_list = data.get('result', {}).get('data', [])
                    if news_list:
                        result = f"今日{category}新闻：\n\n"
                        for i, news in enumerate(news_list[:count], 1):
                            result += f"{i}. {news.get('title', '')}\n"
                            result += f"   来源：{news.get('author_name', '未知')}\n"
                            result += f"   链接：{news.get('url', '')}\n"
                            result += f"   发布时间：{news.get('date', '')}\n\n"
                        return result
                    else:
                        return f"未找到{category}类别的新闻"
                else:
                    return f"获取新闻失败：{data.get('reason', '未知错误')}"
            else:
                return f"请求失败：HTTP状态码 {response.status_code}"
        except Exception as e:
            return f"获取新闻信息出错：{str(e)}"

# ====== 初始化助手服务 ======
def init_agent_service():
    """初始化助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='智能助手',
            description='智能助手，支持天气查询和新闻获取',
            system_message="你是一名有用的助手",
            function_list=['get_current_weather', 'get_news'],  # 增加新闻工具
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式
    
    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input('user question: ')
                # 获取可选的文件输入
                file = input('file url (press enter if no file): ').strip()
                
                # 输入验证
                if not query:
                    print('user question cannot be empty！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举3个典型门票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '北京今天的天气怎么样？',
                '获取今日头条新闻'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 运行模式选择
    app_gui()          # 图形界面模式（默认）
    # app_tui()