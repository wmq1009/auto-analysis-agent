#!/usr/bin/env python3
# 测试修改后的LLMParser类，特别是qwen3模型的连接功能

import os
from modules.llm_parser import LLMParser

# 测试LLMParser类的连接功能
def test_llm_parser_connection():
    """
    测试LLMParser类的连接功能
    """
    print("测试修改后的LLMParser类的连接功能...")
    
    # 从环境变量获取API密钥（如果存在）
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("请先设置OPENAI_API_KEY环境变量")
        return
    
    # 测试qwen3模型
    print("\n测试qwen3模型...")
    llm_parser = LLMParser(api_key=api_key, model="qwen3")
    print(f"模型名称: {llm_parser.model}")
    print(f"实际使用的模型名称: {llm_parser.actual_model}")
    print(f"客户端类型: {llm_parser.client_type}")
    success, message = llm_parser.test_connection()
    print(f"连接结果: {'成功' if success else '失败'}")
    print(f"消息: {message}")
    
    # 测试qwen3-large模型
    print("\n测试qwen3-large模型...")
    llm_parser = LLMParser(api_key=api_key, model="qwen3-large")
    print(f"模型名称: {llm_parser.model}")
    print(f"实际使用的模型名称: {llm_parser.actual_model}")
    print(f"客户端类型: {llm_parser.client_type}")
    success, message = llm_parser.test_connection()
    print(f"连接结果: {'成功' if success else '失败'}")
    print(f"消息: {message}")
    
    # 测试qwen3-coder模型
    print("\n测试qwen3-coder模型...")
    llm_parser = LLMParser(api_key=api_key, model="qwen3-coder")
    print(f"模型名称: {llm_parser.model}")
    print(f"实际使用的模型名称: {llm_parser.actual_model}")
    print(f"客户端类型: {llm_parser.client_type}")
    success, message = llm_parser.test_connection()
    print(f"连接结果: {'成功' if success else '失败'}")
    print(f"消息: {message}")

if __name__ == "__main__":
    test_llm_parser_connection()
