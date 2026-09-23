import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load('titanic_model.pkl')
df = pd.read_csv('data_clean.csv')

st.title("🚢 Titanic Survival Predictor")
st.write("Prediksi apakah penumpang akan selamat")

tab1, tab2 = st.tabs(["🔮 Predict", "📊 Data Visualization"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox("Pclass", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
    with col2:
        age = st.number_input("Age", 1, 100, 30)
        fare = st.number_input("Fare", 0.0, 500.0, 50.0)

    if st.button("Predict Now"):
        sex_num = 0 if sex == "male" else 1
        input_df = pd.DataFrame([[pclass, sex_num, age, fare]], columns=['Pclass', 'Sex', 'Age', 'Fare'])
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ SURVIVED! Probability: {prob:.2f}")
        else:
            st.error(f"❌ NOT SURVIVED. Probability: {1-prob:.2f}")

with tab2:
    st.subheader("Data Insights")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x='Sex', y='Survived', data=df, ax=ax[0])
    ax[0].set_title('Survival by Sex')
    sns.barplot(x='Pclass', y='Survived', data=df, ax=ax[1])
    ax[1].set_title('Survival by Pclass')
    st.pyplot(fig)
