#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试代码修复功能的脚本
"""

import pandas as pd
import numpy as np
from modules.code_generator import CodeGenerator
from modules.llm_parser import LLMParser

# 创建测试数据
data = {
    '年龄': [25, 30, 35, 40, 45, 50, 55, 60],
    '性别': ['男', '女', '男', '女', '男', '女', '男', '女'],
    '血压': [120, 130, 140, 150, 160, 170, 180, 190],
    '血糖': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
}
df = pd.DataFrame(data)
data_types = {
    '年龄': '连续型数值',
    '性别': '分类变量',
    '血压': '连续型数值',
    '血糖': '连续型数值'
}

# 初始化代码生成器
code_gen = CodeGenerator()

print("=== 测试1: 修复未导入的库 ===")
# 故意生成缺少导入的代码
buggy_code = """
# 分析年龄的分布
# 缺少导入numpy

data = df['年龄'].dropna()

# 计算统计量
mean_age = np.mean(data)
median_age = np.median(data)
std_age = np.std(data)

print(f'平均年龄: {mean_age}')
print(f'中位年龄: {median_age}')
print(f'年龄标准差: {std_age}')

# 保存结果
result = {'mean': mean_age, 'median': median_age, 'std': std_age}
"""

# 模拟运行代码产生错误
try:
    exec_globals = {"df": df}
    exec(buggy_code, exec_globals)
except Exception as e:
    error_msg = str(e)
    print(f"原始代码错误: {error_msg}")
    
    # 测试修复功能
    fixed_code = code_gen.fix_code(buggy_code, error_msg, df, data_types)
    print("\n修复后的代码:")
    print(fixed_code)
    
    # 测试修复后的代码
    try:
        exec_globals = {"df": df}
        exec(fixed_code, exec_globals)
        print("\n修复后的代码执行成功!")
    except Exception as e2:
        print(f"\n修复后的代码仍然出错: {str(e2)}")

print("\n" + "="*50)
print("=== 测试2: 修复数据类型错误 ===")
# 故意生成数据类型错误的代码
buggy_code2 = """
# 计算男女平均年龄差异
# 错误的列名

male_ages = df['男'].dropna()
female_ages = df['女'].dropna()

mean_male = male_ages.mean()
mean_female = female_ages.mean()

difference = mean_male - mean_female

print(f'男性平均年龄: {mean_male}')
print(f'女性平均年龄: {mean_female}')
print(f'男女平均年龄差异: {difference}')

result = {'male_mean': mean_male, 'female_mean': mean_female, 'difference': difference}
"""

# 模拟运行代码产生错误
try:
    exec_globals = {"df": df}
    exec(buggy_code2, exec_globals)
except Exception as e:
    error_msg = str(e)
    print(f"原始代码错误: {error_msg}")
    
    # 测试修复功能
    fixed_code = code_gen.fix_code(buggy_code2, error_msg, df, data_types)
    print("\n修复后的代码:")
    print(fixed_code)
    
    # 测试修复后的代码
    try:
        exec_globals = {"df": df}
        exec(fixed_code, exec_globals)
        print("\n修复后的代码执行成功!")
    except Exception as e2:
        print(f"\n修复后的代码仍然出错: {str(e2)}")

print("\n" + "="*50)
print("=== 测试3: 迭代修复机制 ===")
# 故意生成多个错误的代码
buggy_code3 = """
# 分析血压和血糖的相关性
# 缺少多个导入

# 提取数据
bp = df['血压']
glu = df['血糖']

# 计算相关系数
corr_coef, p_value = stats.pearsonr(bp, glu)

# 绘制散点图
plt.scatter(bp, glu)
plt.title('血压与血糖的相关性')
plt.xlabel('血压')
plt.ylabel('血糖')
plt.show()

result = {'correlation': corr_coef, 'p_value': p_value}
"""

# 模拟迭代修复过程
max_retries = 3
iteration = 0
current_code = buggy_code3

while iteration <= max_retries:
    try:
        iteration += 1
        print(f"\n尝试 {iteration}/{max_retries+1}:")
        
        exec_globals = {"df": df}
        exec(current_code, exec_globals)
        print("代码执行成功!")
        break
        
    except Exception as e:
        error_msg = str(e)
        print(f"错误: {error_msg}")
        
        if iteration > max_retries:
            print("达到最大重试次数，修复失败")
            break
            
        # 修复代码
        current_code = code_gen.fix_code(current_code, error_msg, df, data_types, iteration)
        print("修复后的代码:")
        print(current_code)
