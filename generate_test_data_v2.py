#!/usr/bin/env python3
# 生成测试数据的脚本 - 版本2
# 包含两种测试数据：
# 1. 连续性结局的重复测量数据
# 2. 二分类的生存数据

import pandas as pd
import numpy as np
import random

# 设置随机种子，确保结果可重现
np.random.seed(42)
random.seed(42)

# =======================================
# 生成连续性结局的重复测量数据
# =======================================
def generate_repeated_measures_data():
    """
    生成连续性结局的重复测量数据
    包含多个患者，每个患者有多次测量，包含多种基线指标
    """
    # 患者数量
    n_patients = 100
    # 每个患者的测量次数（随机2-5次）
    n_measurements = np.random.randint(2, 6, size=n_patients)
    
    # 基线指标：性别、年龄、BMI、高血压史、糖尿病史、吸烟史
    patients = []
    for i in range(n_patients):
        patient_id = f"P{i+1:03d}"
        gender = random.choice([0, 1])  # 0=男性, 1=女性
        age = random.randint(40, 80)
        bmi = round(random.uniform(18, 35), 1)
        hypertension = random.choice([0, 1])  # 0=无, 1=有
        diabetes = random.choice([0, 1])  # 0=无, 1=有
        smoking = random.choice([0, 1])  # 0=无, 1=有
        
        # 生成多次测量数据
        for j in range(n_measurements[i]):
            # 测量时间点（基线、1周、2周、1个月、3个月）
            time_point = j
            time_label = ["基线", "1周", "2周", "1个月", "3个月"][j]
            
            # 连续性结局：血压（收缩压）
            # 基线血压根据年龄、BMI、高血压史生成
            base_bp = 110 + 0.5 * age + 2 * bmi + 10 * hypertension
            # 每次测量的血压有波动，且有治疗效果（假设治疗有效，血压逐渐降低）
            bp = base_bp - 2 * j + np.random.normal(0, 5)
            bp = round(max(80, bp), 1)  # 确保血压合理
            
            # 其他测量指标：血糖、胆固醇
            glucose = round(5 + 0.1 * age + 0.2 * bmi + 2 * diabetes + np.random.normal(0, 0.5), 1)
            cholesterol = round(3 + 0.02 * age + 0.1 * bmi + np.random.normal(0, 0.3), 1)
            
            patients.append({
                "patient_id": patient_id,
                "time_point": time_point,
                "time_label": time_label,
                "gender": gender,
                "age": age,
                "bmi": bmi,
                "hypertension": hypertension,
                "diabetes": diabetes,
                "smoking": smoking,
                "systolic_bp": bp,
                "glucose": glucose,
                "cholesterol": cholesterol
            })
    
    df = pd.DataFrame(patients)
    return df

# =======================================
# 生成二分类的生存数据
# =======================================
def generate_survival_data():
    """
    生成二分类的生存数据
    每个患者拥有多天的数据，包含血常规等每天测量值，基线数据和最终结局
    """
    # 患者数量
    n_patients = 100
    
    patients = []
    for i in range(n_patients):
        patient_id = f"P{i+1:03d}"
        
        # 基线数据
        gender = random.choice([0, 1])  # 0=男性, 1=女性
        age = random.randint(40, 80)
        bmi = round(random.uniform(18, 35), 1)
        cancer_stage = random.choice([1, 2, 3, 4])  # 癌症分期
        treatment = random.choice([0, 1])  # 0=常规治疗, 1=新疗法
        
        # 生存天数（随机30-365天）
        survival_days = random.randint(30, 365)
        
        # 结局：1=死亡, 0=存活
        outcome = random.choice([0, 1])
        
        # 生成每天的数据
        for day in range(survival_days):
            # 血常规指标：白细胞计数、红细胞计数、血小板计数
            wbc = round(4 + np.random.normal(0, 1), 1)  # 白细胞计数 (4-10 ×10^9/L)
            rbc = round(4.5 + np.random.normal(0, 0.5), 2)  # 红细胞计数 (4.0-5.5 ×10^12/L)
            platelets = round(250 + np.random.normal(0, 50), 0)  # 血小板计数 (150-400 ×10^9/L)
            
            # 其他指标：C反应蛋白、白蛋白
            crp = round(5 + np.random.normal(2, 3), 1)  # C反应蛋白 (0-10 mg/L)
            albumin = round(40 + np.random.normal(2, 1), 1)  # 白蛋白 (35-50 g/L)
            
            # 结局变量：只有最后一天可能为1，之前都是0
            day_outcome = outcome if day == survival_days - 1 else 0
            
            patients.append({
                "patient_id": patient_id,
                "day": day + 1,  # 天数从1开始
                "gender": gender,
                "age": age,
                "bmi": bmi,
                "cancer_stage": cancer_stage,
                "treatment": treatment,
                "wbc": wbc,
                "rbc": rbc,
                "platelets": platelets,
                "crp": crp,
                "albumin": albumin,
                "outcome": day_outcome
            })
    
    df = pd.DataFrame(patients)
    return df

# =======================================
# 主函数
# =======================================
if __name__ == "__main__":
    # 生成连续性结局的重复测量数据
    print("生成连续性结局的重复测量数据...")
    repeated_measures_df = generate_repeated_measures_data()
    repeated_measures_df.to_csv("repeated_measures_data.csv", index=False, encoding="utf-8-sig")
    print(f"连续性结局的重复测量数据生成完成，共 {len(repeated_measures_df)} 行，已保存到 repeated_measures_data.csv")
    
    # 生成二分类的生存数据
    print("\n生成二分类的生存数据...")
    survival_df = generate_survival_data()
    survival_df.to_csv("survival_data.csv", index=False, encoding="utf-8-sig")
    print(f"二分类的生存数据生成完成，共 {len(survival_df)} 行，已保存到 survival_data.csv")
    
    print("\n所有测试数据生成完成！")
