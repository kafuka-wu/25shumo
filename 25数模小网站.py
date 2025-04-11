import base64
import streamlit as st
import pandas as pd
import joblib

# 加载模型
models = {
    'XGBoost': joblib.load('optimized_model.pkl')  # Windows 示例
}

# 定义输入字段的映射
def get_input_fields():
    # 添加体重和身高输入
    weight = st.number_input("您的体重是多少？（单位：千克）", min_value=0.0, max_value=500.0, value=60.0, step=0.1)
    height = st.number_input("您的身高是多少？（单位：米）", min_value=0.0, max_value=3.0, value=1.7, step=0.01)
    # 计算BMI
    bmi = weight / (height ** 2)
    st.write(f"您的BMI是：{bmi:.2f}")

    return {
        'RIAGENDR': st.selectbox("性别", ['男', '女']),
        'RIDAGEYR': st.slider("年龄", min_value=0, max_value=100, value=50),
        'DMDEDUC2': st.selectbox("您的教育水平如何？",
                                 ['九年级以下', '九到十一年级（包括十二年级，但无文凭）', '高中毕业/GED或同等学历',
                                  '一些大学或AA学位', '大专以上学历']),
        'BMXBMI': bmi,  # 使用计算出的BMI
        'BMXARMC': st.slider("您的上臂围是多少？（单位：cm）", min_value=0.0, max_value=100.0, value=30.0, step=0.1),
        'LBXTC': st.slider("您的总胆固醇是多少？（单位：mg/dL）", min_value=0.0, max_value=400.0, value=150.0, step=0.1),
        'LBDHDD': st.slider("您的HDL-胆固醇是多少？（单位：mg/dL）", min_value=0.0, max_value=100.0, value=40.0, step=0.1),
        'LBXPLTSI': st.slider("您的血小板计数是多少？（单位：1000 个细胞/μL）", min_value=0.0, max_value=1000.0, value=250.0, step=0.1),
        'LBXWBCSI': st.slider("您的白细胞计数是多少？（单位：1000 个细胞/μL）", min_value=0.0, max_value=20.0, value=7.0, step=0.1),
        'LBXBPB': st.slider("您的血铅值是多少？（单位：ug/dL）", min_value=0.0, max_value=100.0, value=10.0, step=0.1),
        'LUXCAPM': st.slider("您的中值 CAP值是多少？（单位：dB/m）", min_value=0.0, max_value=100.0, value=50.0, step=0.1),
        'LBXTHG': st.slider("您的总血汞是多少？（单位：ug/L）", min_value=0.0, max_value=100.0, value=10.0, step=0.1),
        'LBXGLU': st.slider("您的血糖是多少？（单位：mg/dL）", min_value=0.0, max_value=200.0, value=100.0, step=0.1),
        'LBXHGB': st.slider("您的血红蛋白是多少？（单位：g/dL）", min_value=0.0, max_value=20.0, value=14.0, step=0.1),
        'LBXGH': st.slider("您的糖化血红蛋白是多少？（单位：%）", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
    }

# 将输入转换为模型所需的格式
def preprocess_input(data):
    processed_data = {}
    for key, value in data.items():
        if isinstance(value, str):  # 如果字段是分类的
            mapping = {
                '男': 0, '女': 1,
                '九年级以下': 0, '九到十一年级（包括十二年级，但无文凭）': 1,
                '高中毕业/GED或同等学历': 2, '一些大学或AA学位': 3, '大专以上学历': 4
            }
            processed_data[key] = mapping[value]
        else:  # 如果字段是数字的
            processed_data[key] = value
    df = pd.DataFrame([processed_data])
    # 重新排序特征，确保与训练模型时的顺序一致
    df = df[['BMXBMI', 'BMXARMC', 'RIAGENDR', 'LBXTHG', 'RIDAGEYR', 'LBDHDD', 'LBXGH', 'LBXHGB', 'DMDEDUC2', 'LUXCAPM', 'LBXWBCSI', 'LBXBPB', 'LBXGLU', 'LBXPLTSI', 'LBXTC']]
    return df

# 预测函数
def predict(data):
    model = models['XGBoost']  # 修复模型键
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1][0] if hasattr(model, "predict_proba") else None
    return prediction[0], probability

# 读取静态图片并转换为 Base64 编码
with open('微信图片_20250329192221.jpg', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# 设置静态图片背景
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpeg;base64,{encoded_string}');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit 应用
st.title("Prediction model of liver fibrosis")
st.markdown("☁☁🌈欢迎来到健康小站！☁☁")
st.markdown("🎇这是肝纤维化风险预测模型！🎇")
st.markdown("是由祖国の小迷弟开发的在线网站，用于帮助有需要的人预测自己是否有肝纤维化的风险。")
st.markdown("请您根据以下的提示选出符合自身条件的选项，最后点击“预测”，即能出现是否具有肝纤维化的结果。")

# 获取用户输入
input_fields = get_input_fields()
data = {key: value for key, value in input_fields.items()}

# 预测按钮
if st.button("预测"):
    prediction, probability = predict(data)
    st.write(f"预测结果: {prediction}")
    st.write(f"概率: {probability:.2f}")

# 在教育水平选择框下面添加解释
st.markdown("""
**教育水平说明：**
- **九年级以下**：未完成九年级教育
- **九到十一年级（包括十二年级，但无文凭）**：完成九到十一年级，或完成十二年级但未获得高中文凭
- **高中毕业/GED或同等学历**：完成高中教育或获得普通教育发展证书（GED）
                        - GED：普通教育发展证书 (General Educational Development) 是为验证个人是否拥有美国或加拿大高中级别学术技能而设立的考试及证书
- **一些大学或AA学位**：完成部分大学课程或获得文学副学士学位（AA）
                   - AA：Associate of Arts，文学副学士学位。
- **大专以上学历**：完成大专或更高学历。
""")

# 结束语
st.markdown("结果解读：")
st.markdown("预测结果为1，则代表您已经是具有肝纤维化的状态；预测结果为0，则代表您目前不具有肝纤维化。概率的多少（0-1）代表您有多大可能性患肝纤维化。")
st.markdown("如果您觉得这个网站能帮到您及有需要的人的话，请帮忙转发，希望能帮助到更多需要的人！😀😀")