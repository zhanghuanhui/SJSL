# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:56:38 2025

@author: admin
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

#加载训练好的随机森林模型(rf.pkl)
model = joblib.load('rf.pkl')


################定义特征名称，对应数据集中自变量的列名
feature_names = [
    "直径", "BC", "noduls", "sex", "AGE",
    "ALP", "CHE", "APOB", "APTT",
    "TGAB", "CEA"
 ]



#### Streamlit 用户界面
st.title("甲状腺转移预测")  #设置网页标题

# 年龄: 数值输入框
AGE = st.number_input("年龄:", min_value=1, max_value=120, value=50)

# 性别: 二分类选择框（0=女性，1=男性）
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# 直径: 数值输入框
直径 = st.number_input("直径:", min_value=0.1, max_value=10.0, value=1.0)
# ALP: 数值输入框
ALP = st.number_input("ALP:", min_value=10, max_value=800, value=100)
# CHE: 数值输入框
CHE = st.number_input("CHE:", min_value=500, max_value=20000, value=8000)
# APOB: 数值输入框
APOB = st.number_input("APOB:", min_value=0.1, max_value=3.0, value=1.2)
# APTT: 数值输入框
APTT = st.number_input("APTT:", min_value=10, max_value=100, value=35)
# TGAB: 数值输入框
TGAB = st.number_input("TGAB:", min_value=0, max_value=2000, value=25)
# CEA: 数值输入框
CEA = st.number_input("CEA:", min_value=0.1, max_value=100.0, value=10.0)


# 多分类选择框定义分类选项  BC= TI-RADS 分级
BC_options = {
    1: 'TI-RADS 1-3级',
    2: 'TI-RADS 4a级',
    3: 'TI-RADS 4b级',
    4: 'TI-RADS 4c级',
    5: 'TI-RADS 5-6级',
}
noduls_options = {
    1: '结节个数 1个',
    2: '结节个数 2个',
    3: '结节个数 >=3个',
}
# BC: 分类选择框
BC = st.selectbox("TI-RADS 分级:", options=list(BC_options.keys()), format_func=lambda x: BC_options[x])
# noduls: 分类选择框
noduls = st.selectbox("结节数量 分类:", options=list(noduls_options.keys()), format_func=lambda x: noduls_options[x])

# 处理输入数据并做出预测
feature_values = [直径, BC, noduls, sex, AGE, ALP, CHE, APOB, APTT, TGAB, CEA]
features = np.array([feature_values])



#当用户点击“Predict”按钮时执行以下代码
if st.button("Predict"):
    #预测类别(0:无转移，1:有转移)
    predicted_class = model.predict(features)[0]
    #预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    #显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    #如果预测类别为1（高风险）
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，你有很高的甲状腺肿瘤转移风险. "
            f"模型预测你发生转移风险的概率是 {probability:.1f}%. "
            "虽然这只是个估计，但它表明你可能面临很大的风险. "
            "我建议你尽快咨询甲状腺方面的专家进行进一步评估和"
            "以确保你获得准确的诊断和必要的治疗."
        )
    #如果预测类别为0（低风险）
    else:
        advice = (
            f"根据我们的模型，你有较低的甲状腺肿瘤转移风险. "
            f"模型预测未发生转移的概率是 {probability:.1f}%. "
            "然而，保持健康的生活方式仍然非常重要. "
            "我建议定期检查以监测您的甲状腺结节."
            "如果您出现任何症状，请及时寻求医疗建议."
        )
    #显示建议
    st.write(advice)

   #SHAP解释
    st.subheader("SHAP ForcePlotExplanation")
    #创建SHAP解释器，基于树模型（如随机森林）
    explainer_shap=shap.TreeExplainer(model)
    #计算SHAP值，用于解释模型的预测
    shap_values=explainer_shap.shap_values(pd.DataFrame([feature_values],columns=feature_names))
    
    #根据预测类别显示SHAP强制图
    #期望值（基线值）
    #解释类别1（患病）的SHAP值
    #特征值数据
    #使用MatpLotLib绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1],shap_values[:,:,1],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)
    #期望值（基线值）
    #解释类别（未患病）的SHAP值
    #特征值数据
    #使用MatpLotLib绘图
    else:
        shap.force_plot(explainer_shap.expected_value[0],shap_values[:,:,0],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
