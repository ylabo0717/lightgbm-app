import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit_shap import st_shap

def main():
    st.title('LightGBM App')

    # load dataset from file dialog
    uploaded_file = st.file_uploader('Upload a dataset', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=0)

        # dropdown list for target variable
        target_variable = st.selectbox('Select the target variable', df.columns)

        x = df.drop(target_variable, axis=1)
        y = df[target_variable]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        lgb_train = lgb.Dataset(x_train, y_train)

        params = {
            'objective': 'mse',
            'num_leaves': 5,
            'seed': 0,
            'verbose': -1
        }

        # train LightGBM model
        model = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=[lgb_train], valid_names=['train'], callbacks=[lgb.log_evaluation(10)])

        st.write('SHAP summary plot')
        explainer = shap.TreeExplainer(model=model, feature_perturbation='tree_path_dependent')
        shap_values = explainer(x_test)
        st_shap(shap.plots.bar(shap_values), height=300)

if __name__ == '__main__':
    main()
