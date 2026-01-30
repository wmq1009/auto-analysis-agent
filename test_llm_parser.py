#!/usr/bin/env python3
# 测试LLMParser类的功能

import os
from modules.llm_parser import LLMParser

# 测试LLMParser类的连接功能
def test_llm_parser_connection():
    """
    测试LLMParser类的连接功能
    """
    print("测试LLMParser类的连接功能...")
    
    # 从环境变量获取API密钥（如果存在）
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("请先设置OPENAI_API_KEY环境变量")
        return
    
    # 测试qwen3模型
    print("\n测试qwen3模型...")
    llm_parser = LLMParser(api_key=api_key, model="qwen3")
    success, message = llm_parser.test_connection()
    print(f"连接结果: {'成功' if success else '失败'}")
    print(f"消息: {message}")
    
    # 测试gpt模型（如果有）
    print("\n测试gpt-4o-mini模型...")
    llm_parser = LLMParser(api_key=api_key, model="gpt-4o-mini")
    success, message = llm_parser.test_connection()
    print(f"连接结果: {'成功' if success else '失败'}")
    print(f"消息: {message}")

if __name__ == "__main__":
    test_llm_parser_connection()
