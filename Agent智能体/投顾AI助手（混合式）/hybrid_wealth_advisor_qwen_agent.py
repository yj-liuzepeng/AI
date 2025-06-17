#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合智能体（Hybrid Agent）- 财富管理投顾AI助手 (Qwen-Agent版)

使用Qwen-Agent框架实现的混合型智能体，结合反应式架构的即时响应能力和深思熟虑架构的长期规划能力，
通过协调层动态切换处理模式，提供智能化财富管理咨询服务。

功能特点：
1. 反应式处理：即时响应客户查询，提供快速反馈
2. 深思熟虑处理：进行复杂的投资分析和长期财务规划
3. 多工具调用：行情查询、资产分析、投资组合优化等
4. 个性化建议：基于客户画像生成定制化财富管理方案
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.gui import WebUI

# 配置 DashScope
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-882e296067b744289acf27e6e20f3ec0')
dashscope.api_key = DASHSCOPE_API_KEY
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 定义系统提示词 ======
SYSTEM_PROMPT = """你是一个专业的财富管理投顾AI助手，可以同时处理简单的快速查询和复杂的财务分析。

作为混合型智能体，你可以：
1. 反应式处理：提供市场行情、基本投资知识、产品信息等快速查询
2. 深思熟虑处理：进行复杂的投资组合分析、风险评估、财务规划等

基于用户查询的复杂度和紧急程度，你会自动选择合适的处理模式。

在回答时，你应该：
- 根据客户风险偏好和投资目标给出个性化建议
- 清晰说明投资建议的依据和潜在风险
- 使用专业但易懂的语言解释复杂概念
- 在不确定时，明确表示局限性并建议咨询人工顾问

你可以使用多种工具进行行情查询、资产分析和投资规划。
"""

# ====== 工具实现 ======

@register_tool('query_market_index')
class MarketIndexTool(BaseTool):
    """
    查询主要市场指数工具
    """
    description = '查询主要市场指数的最新行情数据'
    parameters = [{
        'name': 'index_name',
        'type': 'string',
        'description': '市场指数名称，例如"上证指数"、"深证成指"、"沪深300"、"创业板指"、"纳斯达克"、"道琼斯"',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        index_name = args['index_name']
        
        # 模拟指数数据（实际应用中应当调用真实API）
        indices = {
            "上证指数": {"price": "3125.62", "change": "+6.32", "pct": "+0.20%"},
            "深证成指": {"price": "9876.54", "change": "-23.45", "pct": "-0.24%"},
            "沪深300": {"price": "3651.82", "change": "+12.56", "pct": "+0.34%"},
            "创业板指": {"price": "2145.67", "change": "-18.32", "pct": "-0.85%"},
            "纳斯达克": {"price": "16432.78", "change": "+145.23", "pct": "+0.89%"},
            "道琼斯": {"price": "38273.45", "change": "-62.34", "pct": "-0.16%"}
        }
        
        if index_name in indices:
            data = indices[index_name]
            return f"{index_name}当前点位: {data['price']}，涨跌: {data['change']}，涨跌幅: {data['pct']}（模拟数据）"
        else:
            return f"未找到 {index_name} 的行情数据。支持的指数包括：上证指数、深证成指、沪深300、创业板指、纳斯达克、道琼斯。"

@register_tool('analyze_portfolio')
class PortfolioAnalysisTool(BaseTool):
    """
    投资组合分析工具
    """
    description = '分析投资组合的风险收益特征、资产配置、行业分布等'
    parameters = [{
        'name': 'portfolio',
        'type': 'object',
        'description': '投资组合数据，包含各资产类别的配置比例',
        'required': True
    }, {
        'name': 'risk_tolerance',
        'type': 'string',
        'description': '客户风险承受能力，如"保守型"、"稳健型"、"平衡型"、"成长型"、"进取型"',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        portfolio = args['portfolio']
        risk_tolerance = args['risk_tolerance']
        
        # 风险评级映射
        risk_mapping = {
            "保守型": 1, "稳健型": 2, "平衡型": 3, "成长型": 4, "进取型": 5
        }
        
        # 资产类别风险等级映射
        asset_risk = {
            "现金": 1, "货币基金": 1,
            "债券": 2, "债券基金": 2,
            "混合基金": 3, "REITs": 3,
            "股票": 4, "股票基金": 4,
            "期货": 5, "期权": 5, "衍生品": 5
        }
        
        # 计算组合风险等级
        total_weight = 0
        weighted_risk = 0
        for asset, weight in portfolio.items():
            if asset in asset_risk:
                weighted_risk += asset_risk[asset] * weight
                total_weight += weight
        
        if total_weight > 0:
            portfolio_risk_score = weighted_risk / total_weight
        else:
            portfolio_risk_score = 0
        
        # 风险等级评定
        risk_levels = ["极低", "低", "中等", "较高", "高"]
        portfolio_risk_level = risk_levels[min(int(portfolio_risk_score) - 1, 4)]
        
        # 风险匹配度评估
        client_risk_score = risk_mapping.get(risk_tolerance, 3)
        risk_diff = portfolio_risk_score - client_risk_score
        
        # 生成分析报告
        analysis = f"投资组合风险等级: {portfolio_risk_level} (风险分: {portfolio_risk_score:.2f})\n"
        analysis += f"客户风险承受能力: {risk_tolerance}\n\n"
        
        if abs(risk_diff) <= 0.5:
            analysis += "风险匹配度: 良好 - 投资组合风险与客户风险承受能力相匹配\n"
        elif risk_diff > 0.5:
            analysis += "风险匹配度: 警告 - 投资组合风险可能高于客户风险承受能力\n"
        else:
            analysis += "风险匹配度: 保守 - 投资组合风险低于客户风险承受能力，可能存在收益不足\n"
        
        # 资产配置分析
        analysis += "\n资产配置分析:\n"
        for asset, weight in portfolio.items():
            risk_text = risk_levels[min(asset_risk.get(asset, 3) - 1, 4)] if asset in asset_risk else "未知"
            analysis += f"- {asset}: {weight*100:.1f}% (风险: {risk_text})\n"
        
        return analysis

@register_tool('optimize_allocation')
class AllocationOptimizerTool(BaseTool):
    """
    资产配置优化工具
    """
    description = '根据客户风险偏好和投资目标，优化资产配置方案'
    parameters = [{
        'name': 'risk_tolerance',
        'type': 'string',
        'description': '客户风险承受能力，如"保守型"、"稳健型"、"平衡型"、"成长型"、"进取型"',
        'required': True
    }, {
        'name': 'investment_horizon',
        'type': 'string',
        'description': '投资期限，如"短期"、"中期"、"长期"',
        'required': True
    }, {
        'name': 'financial_goals',
        'type': 'array',
        'description': '财务目标列表',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        args = json.loads(params)
        risk_tolerance = args['risk_tolerance']
        investment_horizon = args['investment_horizon']
        financial_goals = args['financial_goals']
        
        # 基于风险偏好的基础配置模板
        base_allocations = {
            "保守型": {"现金": 0.15, "货币基金": 0.25, "债券": 0.40, "债券基金": 0.10, "股票": 0.05, "股票基金": 0.05},
            "稳健型": {"现金": 0.10, "货币基金": 0.15, "债券": 0.30, "债券基金": 0.15, "股票": 0.15, "股票基金": 0.15},
            "平衡型": {"现金": 0.05, "货币基金": 0.10, "债券": 0.25, "债券基金": 0.15, "股票": 0.25, "股票基金": 0.20},
            "成长型": {"现金": 0.05, "货币基金": 0.05, "债券": 0.15, "债券基金": 0.15, "股票": 0.35, "股票基金": 0.25},
            "进取型": {"现金": 0.05, "货币基金": 0.05, "债券": 0.05, "债券基金": 0.10, "股票": 0.40, "股票基金": 0.35}
        }
        
        # 根据投资期限调整
        horizon_adjustments = {
            "短期": {"现金": 0.10, "货币基金": 0.05, "债券": 0.05, "债券基金": 0, "股票": -0.10, "股票基金": -0.10},
            "中期": {"现金": 0, "货币基金": 0, "债券": 0, "债券基金": 0, "股票": 0, "股票基金": 0},
            "长期": {"现金": -0.05, "货币基金": -0.05, "债券": -0.05, "债券基金": 0, "股票": 0.10, "股票基金": 0.05}
        }
        
        # 获取基础配置
        allocation = base_allocations.get(risk_tolerance, base_allocations["平衡型"]).copy()
        
        # 应用期限调整
        horizon_adj = horizon_adjustments.get(investment_horizon, horizon_adjustments["中期"])
        for asset, adj in horizon_adj.items():
            if asset in allocation:
                allocation[asset] = max(0, allocation[asset] + adj)
        
        # 应用特定财务目标的调整
        goal_keywords = [goal.lower() for goal in financial_goals]
        
        # 检查是否有特定目标并进行调整
        if any("退休" in goal for goal in goal_keywords):
            if investment_horizon == "长期":
                allocation["股票基金"] = min(0.45, allocation["股票基金"] + 0.05)
                allocation["REITs"] = allocation.get("REITs", 0) + 0.05
        
        if any("教育" in goal for goal in goal_keywords):
            allocation["债券"] = min(0.45, allocation["债券"] + 0.05)
            allocation["股票基金"] = max(0, allocation["股票基金"] - 0.05)
        
        if any("购房" in goal for goal in goal_keywords):
            allocation["债券"] = min(0.50, allocation["债券"] + 0.10)
            allocation["现金"] = min(0.25, allocation["现金"] + 0.05)
            allocation["股票"] = max(0, allocation["股票"] - 0.10)
        
        # 归一化分配比例
        total = sum(allocation.values())
        normalized_allocation = {k: v/total for k, v in allocation.items()}
        
        # 生成配置建议
        result = f"为{risk_tolerance}风险偏好、{investment_horizon}投资期限的客户定制的资产配置方案：\n\n"
        for asset, weight in normalized_allocation.items():
            result += f"- {asset}: {weight*100:.1f}%\n"
        
        # 添加预期收益率和风险区间（模拟值）
        expected_returns = {
            "保守型": [2.0, 4.0], "稳健型": [3.0, 6.0], "平衡型": [4.0, 8.0], 
            "成长型": [6.0, 12.0], "进取型": [8.0, 15.0]
        }
        
        returns = expected_returns.get(risk_tolerance, expected_returns["平衡型"])
        result += f"\n预期年化收益率区间: {returns[0]}%-{returns[1]}%\n"
        result += f"建议投资期限: {investment_horizon}\n"
        
        return result

# ====== 工具注册和初始化 ======

def init_wealth_advisor():
    """初始化财富顾问智能体"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 2,
    }
    
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='财富管理投顾助手',
            description='提供个性化财富管理咨询服务',
            system_message=SYSTEM_PROMPT,
            function_list=['query_market_index', 'analyze_portfolio', 'optimize_allocation'],
        )
        print("财富顾问智能体初始化成功！")
        return bot
    except Exception as e:
        print(f"财富顾问智能体初始化失败: {str(e)}")
        raise

# ====== CLI和GUI界面 ======

def app_cli():
    """命令行交互界面"""
    try:
        # 初始化智能体
        advisor = init_wealth_advisor()
        print("=" * 60)
        print("欢迎使用财富管理投顾AI助手 (Qwen-Agent版)")
        print("输入问题开始对话，输入'exit'退出")
        print("=" * 60)
        
        # 对话历史
        messages = []
        
        # 模拟客户画像数据
        default_profile = {
            "customer_id": "user123",
            "risk_tolerance": "平衡型",
            "investment_horizon": "中期",
            "financial_goals": ["稳健增值", "子女教育金"],
            "portfolio": {
                "现金": 0.1,
                "债券": 0.3,
                "股票": 0.4,
                "基金": 0.2
            }
        }
        
        # 打印客户画像
        print("当前客户画像:")
        print(f"- 风险偏好: {default_profile['risk_tolerance']}")
        print(f"- 投资期限: {default_profile['investment_horizon']}")
        print(f"- 财务目标: {', '.join(default_profile['financial_goals'])}")
        print("您可以在咨询中更新这些信息")
        print("-" * 60)
        
        while True:
            try:
                # 获取用户输入
                query = input('请输入您的问题: ')
                
                # 检查退出命令
                if query.lower() in ['exit', 'quit', '退出']:
                    print("感谢使用财富管理投顾AI助手，再见！")
                    break
                
                # 输入验证
                if not query.strip():
                    print('问题不能为空！')
                    continue
                
                # 添加当前客户画像信息到查询
                query_with_profile = f"我的个人情况：风险偏好{default_profile['risk_tolerance']}，" \
                                    f"投资期限{default_profile['investment_horizon']}，" \
                                    f"财务目标是{', '.join(default_profile['financial_goals'])}。\n\n" \
                                    f"我的问题是：{query}"
                
                # 构建消息
                messages.append({'role': 'user', 'content': query_with_profile})
                
                print("\n正在思考...")
                # 运行助手并处理响应
                for chunk in advisor.run_stream(messages):
                    # 打印每个响应片段
                    if isinstance(chunk, dict) and chunk.get('role') == 'assistant':
                        content = chunk.get('content', '')
                        if content:
                            print(content, end='', flush=True)
                
                # 获取完整响应并添加到消息历史
                response = advisor.run(messages)
                messages.extend(response)
                print("\n" + "-" * 60)
                
            except Exception as e:
                print(f"\n处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
                print("-" * 60)
    
    except Exception as e:
        print(f"启动命令行界面失败: {str(e)}")

def app_gui():
    """图形用户界面"""
    try:
        print("正在启动Web界面...")
        # 初始化智能体
        advisor = init_wealth_advisor()
        
        # 配置聊天界面
        chatbot_config = {
            'prompt.suggestions': [
                '请分析我当前的投资组合是否合理？',
                '根据我的风险偏好，如何优化资产配置？',
                '上证指数今天的行情如何？',
                '我想为子女教育储备资金，有哪些合适的投资方案？'
            ]
        }
        
        print("Web界面准备就绪，正在启动服务...")
        # 启动Web界面
        WebUI(
            advisor,
            chatbot_config=chatbot_config
        ).run()
    
    except Exception as e:
        print(f"启动Web界面失败: {str(e)}")
        print("请检查网络连接和API Key配置")

# 主函数
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='财富管理投顾AI助手')
    parser.add_argument('--gui', action='store_true', help='启动图形界面')
    parser.add_argument('--cli', action='store_true', help='启动命令行界面')
    
    args = parser.parse_args()
    
    if args.gui:
        app_gui()
    elif args.cli:
        app_cli()
    else:
        # 默认启动图形界面
        app_gui() 