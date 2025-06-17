# -*- coding: utf-8 -*-
"""
保险场景 VLM使用
依赖：
    pip install openai pandas openpyxl
"""
import os
from openai import OpenAI
import pandas as pd

def get_response(user_prompt, image_url):
    """
    调用VLM，得到推理结果
    user_prompt：用户想要分析的内容
    image_url：想要分析的图片
    """
    # 得到image_url_list，一张图片也放到[]中
    if image_url.startswith('[') and ',' in image_url:
        # 属于image_url list
        image_url = image_url.strip()
        image_url = image_url[1:-1]
        image_url_list = image_url.split(',')
        image_url_list = [temp_url.strip() for temp_url in image_url_list]
    else:
        image_url_list = [image_url]
    # 得到messages
    content = [{"type": "text", "text": f"{user_prompt}"}]
    for temp_url in image_url_list:
        image_url_full = f"https://vl-image.oss-cn-shanghai.aliyuncs.com/{temp_url}.jpg"
        content.append({"type": "image_url", "image_url": {"url": f"{image_url_full}"}})
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    # 从环境变量获取API Key
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not api_key:
        raise ValueError("未检测到环境变量 DASHSCOPE_API_KEY，请先设置你的 DashScope API Key！")
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max",  # qwen-vl-plus
        messages=messages
    )
    return completion

def main():
    # 读取Excel文件
    df = pd.read_excel('./prompt_template_cn.xlsx')
    df['response'] = ''
    for index, row in df.iterrows():
        user_prompt = row['prompt']
        image_url = row['image']
        # 得到VLM推理结果
        completion = get_response(user_prompt, image_url)
        response = completion.choices[0].message.content
        df.loc[index, 'response'] = response
        print(f"{index+1} {user_prompt} {image_url}")
    # 保存结果到Excel
    df.to_excel('./prompt_template_cn_result.xlsx', index=False)

if __name__ == '__main__':
    main() 