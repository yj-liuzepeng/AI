import os
import dashscope
from dashscope import Generation
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI
import json
from typing import Dict, List, Any

# 配置 DashScope
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-882e296067b744289acf27e6e20f3ec0')
dashscope.api_key = DASHSCOPE_API_KEY

# ====== 提示模板 ======
PERCEPTION_PROMPT = """你是一个专业的投资研究分析师，请收集和整理关于以下研究主题的市场数据和信息：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

请从以下几个方面进行市场感知：
1. 市场概况和最新动态
2. 关键经济和市场指标
3. 近期重要新闻（至少3条）
4. 行业趋势分析（至少针对3个细分领域）

根据你的专业知识和经验，提供尽可能详细和准确的信息。

输出格式要求为JSON，包含以下字段：
- market_overview: 字符串
- key_indicators: 字典，键为指标名称，值为指标值和简要解释
- recent_news: 字符串列表，每项为一条重要新闻
- industry_trends: 字典，键为细分领域，值为趋势分析
"""

MODELING_PROMPT = """你是一个资深投资策略师，请根据以下市场数据和信息，构建市场内部模型，进行深度分析：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

请构建一个全面的市场内部模型，包括：
1. 当前市场状态评估
2. 经济周期判断
3. 主要风险因素（至少3个）
4. 潜在机会领域（至少3个）
5. 市场情绪分析

输出格式要求为JSON，包含以下字段：
- market_state: 字符串
- economic_cycle: 字符串
- risk_factors: 字符串列表
- opportunity_areas: 字符串列表
- market_sentiment: 字符串
"""

REASONING_PROMPT = """你是一个战略投资顾问，请根据以下市场模型，生成3个不同的投资分析方案：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

请为每个方案提供：
1. 方案ID（简短标识符）
2. 投资假设
3. 分析方法
4. 预期结果
5. 置信度（0-1之间的小数）
6. 方案优势（至少3点）
7. 方案劣势（至少2点）

这些方案应该有明显的差异，代表不同的投资思路或分析角度。

输出格式要求为JSON数组，每个元素包含以下字段：
- plan_id: 字符串
- hypothesis: 字符串
- analysis_approach: 字符串
- expected_outcome: 字符串
- confidence_level: 浮点数
- pros: 字符串列表
- cons: 字符串列表
"""

DECISION_PROMPT = """你是一个投资决策委员会主席，请评估以下候选分析方案，选择最优方案并形成投资决策：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场内部模型:
{world_model}

候选分析方案:
{reasoning_plans}

请基于方案的假设、分析方法、预期结果、置信度以及优缺点，选择最优的投资方案，并给出详细的决策理由。
你的决策应该综合考虑投资潜力、风险水平和时间框架的匹配度。

输出格式要求为JSON，包含以下字段：
- selected_plan_id: 字符串
- investment_thesis: 字符串
- supporting_evidence: 字符串列表
- risk_assessment: 字符串
- recommendation: 字符串
- timeframe: 字符串
"""

REPORT_PROMPT = """你是一个专业的投资研究报告撰写人，请根据以下信息生成一份完整的投资研究报告：

研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

市场数据和信息:
{perception_data}

市场内部模型:
{world_model}

选定的投资决策:
{selected_plan}

请生成一份结构完整、逻辑清晰的投研报告，包括但不限于：
1. 报告标题和摘要
2. 市场和行业背景
3. 核心投资观点
4. 详细分析论证
5. 风险因素
6. 投资建议
7. 时间框架和预期回报

报告应当专业、客观，同时提供足够的分析深度和洞见。
"""

# ====== 基础LLM调用函数 ======
def call_llm(prompt: str) -> str:
    """调用DashScope的LLM进行推理"""
    response = Generation.call(
        model='qwen-turbo-2025-04-28',
        prompt=prompt,
        result_format='message',
        temperature=0.7,
        top_p=0.8,
        max_tokens=2000,
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"调用LLM失败: {response.message}")

# ====== system prompt ======
system_prompt = """你是深思熟虑型智能投研助手，能够分阶段完成市场感知、建模分析、推理方案生成、决策与报告。请根据用户输入的研究主题、行业焦点和时间范围，主动选择合适的工具，输出专业、系统、结构化的投研结论。"""

# ====== 工具1：市场感知 ======
@register_tool('perception')
class PerceptionTool(BaseTool):
    """
    收集并整理市场数据和信息
    """
    description = '收集并整理市场数据和信息，输入为研究主题、行业焦点、时间范围'
    parameters = [
        {'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True},
        {'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True},
        {'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True},
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        
        # 准备提示
        prompt = PERCEPTION_PROMPT.format(
            research_topic=args['research_topic'],
            industry_focus=args['industry_focus'],
            time_horizon=args['time_horizon']
        )
        
        # 调用LLM
        try:
            result = call_llm(prompt)
            # 验证结果是否为有效的JSON格式
            json_result = json.loads(result)
            return json.dumps(json_result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 如果返回结果不是有效JSON格式，尝试提取JSON部分
            try:
                # 尝试提取花括号中的内容
                import re
                json_content = re.search(r'({.*})', result, re.DOTALL)
                if json_content:
                    json_result = json.loads(json_content.group(1))
                    return json.dumps(json_result, ensure_ascii=False, indent=2)
                else:
                    raise Exception("无法从LLM响应中提取有效JSON")
            except Exception as e:
                return json.dumps({
                    "market_overview": "LLM返回格式错误，无法解析市场概况",
                    "key_indicators": {"错误": "数据解析失败"},
                    "recent_news": ["LLM返回格式错误，无法解析新闻"],
                    "industry_trends": {"错误": "数据解析失败"}
                }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "error": f"调用LLM时发生错误: {str(e)}",
                "market_overview": "调用失败，无法获取市场概况",
                "key_indicators": {"错误": "调用失败"},
                "recent_news": ["调用失败，无法获取新闻"],
                "industry_trends": {"错误": "调用失败"}
            }, ensure_ascii=False, indent=2)

# ====== 工具2：建模分析 ======
@register_tool('modeling')
class ModelingTool(BaseTool):
    """
    构建市场内部模型，分析风险与机会
    """
    description = '根据市场数据构建内部模型，输入为市场数据JSON字符串'
    parameters = [
        {'name': 'perception_data', 'type': 'string', 'description': '市场数据JSON字符串', 'required': True},
        {'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True},
        {'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True},
        {'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True},
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        
        # 准备提示
        prompt = MODELING_PROMPT.format(
            research_topic=args['research_topic'],
            industry_focus=args['industry_focus'],
            time_horizon=args['time_horizon'],
            perception_data=args['perception_data']
        )
        
        # 调用LLM
        try:
            result = call_llm(prompt)
            # 验证结果是否为有效的JSON格式
            json_result = json.loads(result)
            return json.dumps(json_result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 如果返回结果不是有效JSON格式，尝试提取JSON部分
            try:
                import re
                json_content = re.search(r'({.*})', result, re.DOTALL)
                if json_content:
                    json_result = json.loads(json_content.group(1))
                    return json.dumps(json_result, ensure_ascii=False, indent=2)
                else:
                    raise Exception("无法从LLM响应中提取有效JSON")
            except Exception as e:
                return json.dumps({
                    "market_state": "LLM返回格式错误，无法解析市场状态",
                    "economic_cycle": "数据解析失败",
                    "risk_factors": ["数据解析失败"],
                    "opportunity_areas": ["数据解析失败"],
                    "market_sentiment": "数据解析失败"
                }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({
                "error": f"调用LLM时发生错误: {str(e)}",
                "market_state": "调用失败，无法获取市场状态",
                "economic_cycle": "调用失败",
                "risk_factors": ["调用失败"],
                "opportunity_areas": ["调用失败"],
                "market_sentiment": "调用失败"
            }, ensure_ascii=False, indent=2)

# ====== 工具3：推理方案生成 ======
@register_tool('reasoning')
class ReasoningTool(BaseTool):
    """
    生成多个投资分析方案
    """
    description = '根据市场模型生成3个投资分析方案，输入为市场模型JSON字符串'
    parameters = [
        {'name': 'world_model', 'type': 'string', 'description': '市场模型JSON字符串', 'required': True},
        {'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True},
        {'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True},
        {'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True},
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        
        # 准备提示
        prompt = REASONING_PROMPT.format(
            research_topic=args['research_topic'],
            industry_focus=args['industry_focus'],
            time_horizon=args['time_horizon'],
            world_model=args['world_model']
        )
        
        # 调用LLM
        try:
            result = call_llm(prompt)
            # 验证结果是否为有效的JSON格式
            json_result = json.loads(result)
            return json.dumps(json_result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 如果返回结果不是有效JSON格式，尝试提取JSON部分
            try:
                import re
                json_content = re.search(r'(\[.*\])', result, re.DOTALL)
                if json_content:
                    json_result = json.loads(json_content.group(1))
                    return json.dumps(json_result, ensure_ascii=False, indent=2)
                else:
                    raise Exception("无法从LLM响应中提取有效JSON")
            except Exception as e:
                return json.dumps([
                    {
                        "plan_id": "ERROR",
                        "hypothesis": "LLM返回格式错误，无法解析",
                        "analysis_approach": "数据解析失败",
                        "expected_outcome": "数据解析失败",
                        "confidence_level": 0,
                        "pros": ["数据解析失败"],
                        "cons": ["数据解析失败"]
                    }
                ], ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps([
                {
                    "plan_id": "ERROR",
                    "hypothesis": f"调用LLM时发生错误: {str(e)}",
                    "analysis_approach": "调用失败",
                    "expected_outcome": "调用失败",
                    "confidence_level": 0,
                    "pros": ["调用失败"],
                    "cons": ["调用失败"]
                }
            ], ensure_ascii=False, indent=2)

# ====== 工具4：决策与报告 ======
@register_tool('decision_report')
class DecisionReportTool(BaseTool):
    """
    选择最优方案并生成研究报告
    """
    description = '根据候选方案和市场模型，选择最优方案并生成研究报告，输入为方案列表和市场模型JSON字符串'
    parameters = [
        {'name': 'reasoning_plans', 'type': 'string', 'description': '方案列表JSON字符串', 'required': True},
        {'name': 'world_model', 'type': 'string', 'description': '市场模型JSON字符串', 'required': True},
        {'name': 'perception_data', 'type': 'string', 'description': '感知数据JSON字符串', 'required': True},
        {'name': 'research_topic', 'type': 'string', 'description': '研究主题', 'required': True},
        {'name': 'industry_focus', 'type': 'string', 'description': '行业焦点', 'required': True},
        {'name': 'time_horizon', 'type': 'string', 'description': '时间范围', 'required': True},
    ]
    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        
        # 第一步：先生成决策
        decision_prompt = DECISION_PROMPT.format(
            research_topic=args['research_topic'],
            industry_focus=args['industry_focus'],
            time_horizon=args['time_horizon'],
            world_model=args['world_model'],
            reasoning_plans=args['reasoning_plans']
        )
        
        try:
            decision_result = call_llm(decision_prompt)
            # 尝试解析JSON
            try:
                decision_json = json.loads(decision_result)
            except json.JSONDecodeError:
                # 尝试提取JSON部分
                import re
                json_content = re.search(r'({.*})', decision_result, re.DOTALL)
                if json_content:
                    decision_json = json.loads(json_content.group(1))
                else:
                    raise Exception("无法从LLM响应中提取有效决策JSON")
            
            # 第二步：生成完整报告
            report_prompt = REPORT_PROMPT.format(
                research_topic=args['research_topic'],
                industry_focus=args['industry_focus'],
                time_horizon=args['time_horizon'],
                perception_data=args['perception_data'],
                world_model=args['world_model'],
                selected_plan=json.dumps(decision_json, ensure_ascii=False)
            )
            
            report_result = call_llm(report_prompt)
            
            # 返回包含决策和报告的结果
            final_result = {
                "decision": decision_json,
                "report": report_result
            }
            
            return json.dumps(final_result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"决策与报告生成失败: {str(e)}",
                "decision": {
                    "selected_plan_id": "ERROR",
                    "investment_thesis": "处理失败",
                    "supporting_evidence": ["处理失败"],
                    "risk_assessment": "处理失败",
                    "recommendation": "处理失败",
                    "timeframe": "处理失败"
                },
                "report": f"报告生成失败，原因: {str(e)}"
            }, ensure_ascii=False, indent=2)

# ====== 初始化助手 ======
def init_agent_service():
    """初始化深思熟虑型投研助手"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'model_server': 'dashscope',
        'timeout': 60,
        'retry_count': 3,
        'generate_cfg': {
            'top_p': 0.8,
            'temperature': 0.7,
            'max_tokens': 2000,
        }
    }
    bot = Assistant(
        llm=llm_cfg,
        name='深思熟虑型投研助手',
        description='多阶段投研流程与结构化报告生成',
        system_message=system_prompt,
        function_list=['perception', 'modeling', 'reasoning', 'decision_report'],
    )
    print("助手初始化成功！")
    return bot

# ====== 终端交互 ======
def app_tui():
    """终端交互模式"""
    bot = init_agent_service()
    messages = []
    print("欢迎使用深思熟虑型投研助手，输入'退出'结束对话。\n")
    while True:
        try:
            query = input('请输入研究主题、行业、周期（如：新能源,电力,中期）：')
            if query.strip() in ['退出', 'exit', 'quit']:
                print('感谢使用，再见！')
                break
            # 简单分割
            parts = [x.strip() for x in query.split(',')]
            if len(parts) != 3:
                print('请输入格式：主题,行业,周期')
                continue
            messages.append({'role': 'user', 'content': f'研究主题:{parts[0]} 行业:{parts[1]} 周期:{parts[2]}'})
            print("正在处理...")
            response = []
            for resp in bot.run(messages):
                print('助手:', resp)
                response.append({'role': 'assistant', 'content': resp})
            messages.extend(response)
        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("请重试或输入新的问题")

# ====== Web界面 ======
def app_gui():
    """Web 图形界面模式"""
    try:
        print("正在启动 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '新能源,电力,中期',
                '高端制造,机械,长期',
                '数字经济,互联网,短期',
            ],
            'input.placeholder': '请输入研究主题、行业、周期（如：新能源,电力,中期）'
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