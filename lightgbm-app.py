import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split

def main():
    st.title('LightGBM App')
    st.write('Welcome to the LightGBM App!')

    # load dataset from file dialog
    uploaded_file = st.file_uploader('Upload a dataset', type=['csv', 'txt'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=0)
        st.write(df)

        X = df.drop('MEDV', axis=1)
        y = df['MEDV']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        lgb_train = lgb.Dataset(X_train, y_train)

        params = {
            'objective': 'mse',
            'num_leaves': 5,
            'seed': 0,
            'verbose': -1
        }

        model = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=[lgb_train], valid_names=['train'], callbacks=[lgb.log_evaluation(10)])

        explainer = shap.TreeExplainer(model=model, feature_perturbation='tree_path_dependent')
        shap_values = explainer(X_test)
        chart_data = {X_train.columns[f]: np.mean(np.abs(shap_values.values[:, f])) for f in range(X_train.shape[1])}
        st.bar_chart(chart_data)

if __name__ == '__main__':
    main()
