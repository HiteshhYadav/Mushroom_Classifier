import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv('./mushrooms.csv')
        label=LabelEncoder()

        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)   
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list,class_names,x_test,y_test):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()

    df=load_data()
    xtrain, xtest, ytrain, ytest=split(df)
    class_names=["Edible", "Poisonous"]
    st.sidebar.subheader("Choose Classifier")
    classifier=st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Logistic Regression", "Random Forest"))

    if classifier=="Support Vector Machine":
        st.sidebar.subheader("Model Hyperparameters")
        C=st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel=st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma=st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
        metrics=st.sidebar.multiselect("Choose Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine Results")
            model=SVC(C=C, kernel=kernel, gamma=gamma) 
            model.fit(xtrain, ytrain)
            accuracy=model.score(xtest, ytest)
            y_pred=model.predict(xtest)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision_score(ytest, y_pred,labels=class_names):.2f}")
            st.write(f"Recall: {recall_score(ytest, y_pred,labels=class_names):.2f}")
            plot_metrics(metrics,class_names,xtest,ytest)
    
    
    if classifier=="Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C=st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter=st.sidebar.slider("Maximum Number of Iterations", 100, 500, key="max_iter")
        metrics=st.sidebar.multiselect("Choose Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key="metrics_LR")

        if st.sidebar.button("Classify", key="classify_LR"):
            st.subheader("Logistic Regression Results")
            model=LogisticRegression(C=C, max_iter=max_iter) 
            model.fit(xtrain, ytrain)
            accuracy=model.score(xtest, ytest)
            y_pred=model.predict(xtest)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision_score(ytest, y_pred,labels=class_names):.2f}")
            st.write(f"Recall: {recall_score(ytest, y_pred,labels=class_names):.2f}")
            plot_metrics(metrics,class_names,xtest,ytest)


    if classifier=="Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators=st.sidebar.number_input("Number of Trees in the Forest", 100, 5000, step=10, key="n_estimators")
        max_depth=st.sidebar.number_input("Maximum Depth of the Tree", 1, 20, step=1, key="max_depth")
        bootstrap=st.sidebar.radio("Bootstrap Samples when Building Trees", (True, False), key="bootstrap") 
        metrics=st.sidebar.multiselect("Choose Metrics to Plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key="metrics_RF")

        if st.sidebar.button("Classify", key="classify_RF"):
            st.subheader("Random Forest Results")
            model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, random_state=0, n_jobs=-1) 
            model.fit(xtrain, ytrain)
            accuracy=model.score(xtest, ytest)
            y_pred=model.predict(xtest)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision_score(ytest, y_pred,labels=class_names):.2f}")
            st.write(f"Recall: {recall_score(ytest, y_pred,labels=class_names):.2f}")
            plot_metrics(metrics,class_names,xtest,ytest) 


    if st.sidebar.checkbox("Show Dataset",False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)

if __name__ == '__main__':
    main()


