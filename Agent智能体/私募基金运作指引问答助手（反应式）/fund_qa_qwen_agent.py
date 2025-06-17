import os
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI

# 配置 DashScope
# 建议将API Key设置为环境变量
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-882e296067b744289acf27e6e20f3ec0')
dashscope.api_key = DASHSCOPE_API_KEY

def get_fund_rules_db():
    # 模拟私募基金规则数据库
    return [
        {
            "id": "rule001",
            "category": "设立与募集",
            "question": "私募基金的合格投资者标准是什么？",
            "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
        },
        {
            "id": "rule002",
            "category": "设立与募集",
            "question": "私募基金的最低募集规模要求是多少？",
            "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
        },
        {
            "id": "rule014",
            "category": "监管规定",
            "question": "私募基金管理人的风险准备金要求是什么？",
            "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
        }
    ]

# ====== system prompt ======
system_prompt = """你是私募基金运作指引问答助手，能够根据用户问题，检索规则库、分类查询或直接生成专业答案。请根据用户需求选择合适的工具，专业、简明、准确地回答私募基金相关问题。"""

# ====== 工具1：关键词检索 ======
@register_tool('search_by_keywords')
class SearchByKeywordsTool(BaseTool):
    """
    通过关键词检索私募基金规则
    """
    description = '通过关键词检索私募基金规则，输入为关键词字符串'
    parameters = [{
        'name': 'keywords',
        'type': 'string',
        'description': '关键词字符串',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        keywords = args['keywords'].lower()
        db = get_fund_rules_db()
        result = []
        for rule in db:
            if any(kw in (rule['category'] + rule['question']).lower() for kw in keywords.split()):
                result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")
        if not result:
            return "未找到相关规则。"
        return '\n\n'.join(result)

# ====== 工具2：类别检索 ======
@register_tool('search_by_category')
class SearchByCategoryTool(BaseTool):
    """
    根据类别检索私募基金规则
    """
    description = '根据类别检索私募基金规则，输入为类别名称'
    parameters = [{
        'name': 'category',
        'type': 'string',
        'description': '类别名称',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        category = args['category'].lower()
        db = get_fund_rules_db()
        result = []
        for rule in db:
            if category in rule['category'].lower():
                result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")
        if not result:
            return f"未找到类别为 '{category}' 的规则。"
        return '\n\n'.join(result)

# ====== 工具3：直接问答 ======
@register_tool('direct_answer')
class DirectAnswerTool(BaseTool):
    """
    直接回答用户关于私募基金的问题（基于简单匹配或通用回复）
    """
    description = '直接回答用户关于私募基金的问题，输入为完整问题字符串'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '用户完整问题',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        query = args['query'].lower()
        db = get_fund_rules_db()
        for rule in db:
            if query in rule['question'].lower():
                return rule['answer']
        # 未命中则返回通用回复
        return "您好，您的问题暂未在知识库中找到直接答案。建议您咨询专业机构或查阅官方资料。"

# ====== 初始化助手 ======
def init_agent_service():
    """初始化私募基金问答助手"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    bot = Assistant(
        llm=llm_cfg,
        name='私募基金运作指引问答助手',
        description='私募基金规则检索与专业问答',
        system_message=system_prompt,
        function_list=['search_by_keywords', 'search_by_category', 'direct_answer'],
    )
    print("助手初始化成功！")
    return bot

# ====== 终端交互 ======
def app_tui():
    """终端交互模式"""
    bot = init_agent_service()
    messages = []
    print("欢迎使用私募基金运作指引问答助手，输入'退出'结束对话。\n")
    while True:
        try:
            query = input('请输入您的问题：')
            if query.strip() in ['退出', 'exit', 'quit']:
                print('感谢使用，再见！')
                break
            messages.append({'role': 'user', 'content': query})
            print("正在处理...")
            response = []
            for resp in bot.run(messages):
                print('助手:', resp)
            messages.extend(response)
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("请重试或输入新的问题")

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '私募基金的合格投资者标准是什么？',
                '如果我是一个净资产800万元的机构，我能投资私募基金吗？',
                '私募基金可以投资哪些资产类别？',
                '请查询设立与募集相关的规则',
                '私募基金管理人的风险准备金要求有哪些？',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    #app_tui()
    app_gui()
