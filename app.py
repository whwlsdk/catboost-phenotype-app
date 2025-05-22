import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import os
import numpy as np

MODEL_PATHS = {
    "과중 (g)": "saved_models/과중 (g)_catboost_model.cbm",
    "과장 (mm)": "saved_models/과장 (mm)_catboost_model.cbm",
    "과폭 (mm)": "saved_models/과폭 (mm)_catboost_model.cbm",
    "과피두께 (mm)": "saved_models/과피두께 (mm)_catboost_model.cbm",
    "과실경도 (kg)": "saved_models/과실경도 (kg)_catboost_model.cbm",
    "당도 (%)": "saved_models/당도 (%)_catboost_model.cbm"
}

@st.cache_resource
def load_models():
    models = {}
    for trait, path in MODEL_PATHS.items():
        if os.path.exists(path):
            model = CatBoostRegressor()
            model.load_model(path)
            models[trait] = model
        else:
            st.warning(f" 모델 파일 없음: {path}")
    return models

st.title("🍅 토마토 유전형 기반 표현형 예측기")
uploaded_file = st.file_uploader("📂 유전형 CSV 파일 업로드", type=["csv"])

if uploaded_file:
    uploaded_file.seek(0)
    geno_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")

    # 인덱스 설정
    for index_col in ["Genotype", "SampleID", "Unnamed: 0"]:
        if index_col in geno_df.columns:
            geno_df = geno_df.set_index(index_col)
            break
    else:
        st.error("❌ 샘플 ID 열이 'Genotype', 'SampleID', 또는 'Unnamed: 0'여야 합니다.")
        st.stop()

    models = load_models()
    all_predictions = {}
    shap_values_dict = {}

    tab1, tab2 = st.tabs(["📈 예측 결과", "🧬 SHAP 해석"])

    with tab1:
        for trait, model in models.items():
            model_snps = model.feature_names_
            user_snps = geno_df.columns

            missing_snps = [snp for snp in model_snps if snp not in user_snps]
            for snp in missing_snps:
                geno_df[snp] = 0
            X = geno_df[model_snps]

            preds = model.predict(X)
            all_predictions[trait] = preds

            # ✅ 예측 통계 표시
            st.subheader(f"📊 {trait} 예측")
            st.write(f"🔹 예측 평균: {np.mean(preds):.4f}")
            st.write(f"🔹 예측 표준편차: {np.std(preds):.4f}")
            st.write(f"🔸 누락된 SNP 수: {len(missing_snps)} / {len(model_snps)}")

        result_df = pd.DataFrame(all_predictions, index=geno_df.index)
        st.dataframe(result_df)

        csv = result_df.to_csv().encode("utf-8-sig")
        st.download_button("⬇️ 예측 결과 CSV 다운로드", data=csv, file_name="토마토_예측결과.csv")

    with tab2:
        st.subheader("🧬 SHAP Feature 영향도 (상위 20개)")
    
        # 🔍 디버깅용: 현재 모델 키 확인
        st.write("✅ 현재 모델 목록 (models.keys()):", list(models.keys()))

        selected_trait = st.selectbox("🔎 표현형 선택", list(models.keys()))

        # ✅ selected_trait 유효성 검증
        if selected_trait not in models:
            st.error(f"❌ 선택한 표현형 '{selected_trait}' 이(가) 모델 목록에 없습니다.")
        
            # 🔍 디버깅: 문자열 일치 비교
            st.write("🔎 선택된 표현형:", repr(selected_trait))
            st.write("🔎 모델 키 목록:", [repr(k) for k in models.keys()])

            for k in models.keys():
                if selected_trait.strip() == k.strip():
                    st.warning(f"⚠️ strip 일치: '{k}'")
                if selected_trait.replace(" ", "") == k.replace(" ", ""):
                    st.warning(f"⚠️ 공백 제거 일치: '{k}'")
        
            st.stop()

        # ✅ 모델 선택 및 SHAP 계산
        model = models[selected_trait]
        X = geno_df[model.feature_names_]

        shap_values = model.get_feature_importance(Pool(X), type="ShapValues")
        shap_values = shap_values[:, :-1]  # 마지막 열은 bias term
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:20]
        top_features = np.array(model.feature_names_)[top_idx]
        top_shap = mean_abs_shap[top_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top_features[::-1], top_shap[::-1])
        ax.set_title(f"{selected_trait} - SHAP Top 20 Features")
        st.pyplot(fig)
