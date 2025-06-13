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
        round_num = 1  # 轮次计数
        while True:
            try:
                print("\n" + "=" * 40)
                print(f"第 {round_num} 轮对话")
                print("=" * 40)
                # 获取用户输入
                query = input('用户问题：').strip()
                # 获取可选的文件输入
                file = input('文件路径（如无可直接回车）：').strip()
                
                # 输入验证
                if not query:
                    print('用户问题不能为空！')
                    continue
                    
                # 构建消息
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})

                print("\n正在处理您的请求...\n")
                # 运行助手并处理响应
                responses = []
                for response in bot.run(messages):
                    responses.append(response)
                # 结构化输出
                print("助手回复：")
                for resp in responses:
                    # 如果是字典或列表，格式化为 JSON
                    if isinstance(resp, (dict, list)):
                        print(json.dumps(resp, ensure_ascii=False, indent=2))
                    else:
                        print(resp)
                # 用分隔线分隔
                print("-" * 40)
                # 将本轮回复加入历史
                messages.extend(responses)
                round_num += 1
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")
