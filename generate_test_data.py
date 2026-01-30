#!/usr/bin/env python3
# 生成测试数据：用药与心血管疾病结局

import pandas as pd
import numpy as np
import random

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)

def generate_cardiovascular_data(n=100, missing_values=10):
    """
    生成心血管疾病相关的测试数据
    
    参数:
        n: 数据行数
        missing_values: 连续性协变量中缺失值的数量
    
    返回:
        DataFrame: 生成的测试数据
    """
    # 生成患者ID
    patient_id = [f"P{str(i).zfill(3)}" for i in range(1, n+1)]
    
    # 生成是否用药（二分类）
    # 假设30%的患者用药
    medication = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    
    # 生成心血管疾病结局（二分类）
    # 假设用药组发病率为10%，未用药组为20%
    outcome = []
    for med in medication:
        if med == 1:
            # 用药组发病率10%
            outcome.append(np.random.choice([0, 1], p=[0.9, 0.1]))
        else:
            # 未用药组发病率20%
            outcome.append(np.random.choice([0, 1], p=[0.8, 0.2]))
    outcome = np.array(outcome)
    
    # 生成年龄（连续变量，30-80岁）
    age = np.random.randint(30, 81, size=n)
    
    # 生成性别（二分类）
    # 假设50%男性，50%女性
    gender = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    
    # 生成收缩压（连续变量，100-180mmHg）
    systolic_bp = np.random.randint(100, 181, size=n)
    
    # 生成舒张压（连续变量，60-110mmHg）
    diastolic_bp = np.random.randint(60, 111, size=n)
    
    # 生成血糖（连续变量，3.0-10.0mmol/L）
    glucose = np.round(np.random.uniform(3.0, 10.0, size=n), 1)
    
    # 生成总胆固醇（连续变量，3.0-8.0mmol/L）
    total_cholesterol = np.round(np.random.uniform(3.0, 8.0, size=n), 1)
    
    # 生成甘油三酯（连续变量，0.5-5.0mmol/L）
    triglycerides = np.round(np.random.uniform(0.5, 5.0, size=n), 1)
    
    # 生成BMI（连续变量，18.0-35.0）
    bmi = np.round(np.random.uniform(18.0, 35.0, size=n), 1)
    
    # 生成心率（连续变量，60-100次/分钟）
    heart_rate = np.random.randint(60, 101, size=n)
    
    # 生成吸烟史（二分类）
    # 假设30%的患者吸烟
    smoking = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    
    # 创建数据框
    df = pd.DataFrame({
        "patient_id": patient_id,
        "medication": medication,
        "outcome": outcome,
        "age": age,
        "gender": gender,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "total_cholesterol": total_cholesterol,
        "triglycerides": triglycerides,
        "bmi": bmi,
        "heart_rate": heart_rate,
        "smoking": smoking
    })
    
    # 添加缺失值到连续性协变量
    # 选择哪些列添加缺失值（只选择连续性协变量）
    continuous_cols = ["age", "systolic_bp", "diastolic_bp", "glucose", 
                     "total_cholesterol", "triglycerides", "bmi", "heart_rate"]
    
    # 生成缺失值的位置
    missing_positions = []
    for _ in range(missing_values):
        # 随机选择一列
        col = random.choice(continuous_cols)
        # 随机选择一行
        row = random.randint(0, n-1)
        missing_positions.append((row, col))
    
    # 添加缺失值
    for row, col in missing_positions:
        df.loc[row, col] = np.nan
    
    return df

def main():
    """
    主函数
    """
    # 生成数据
    df = generate_cardiovascular_data(n=100, missing_values=10)
    
    # 保存数据到CSV文件
    df.to_csv("test_data.csv", index=False)
    
    # 打印数据信息
    print("生成的数据信息：")
    print(f"数据行数：{len(df)}")
    print(f"数据列数：{len(df.columns)}")
    print("\n数据前5行：")
    print(df.head())
    
    # 打印缺失值信息
    print("\n缺失值统计：")
    print(df.isnull().sum())
    
    print("\n数据已保存到 test_data.csv 文件中")

if __name__ == "__main__":
    main()
