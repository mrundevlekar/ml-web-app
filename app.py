import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



page = st.sidebar.selectbox("Choose a page", ["Upload", "Pre-processing", "Modeling"])

st.title ("Machine Learning Model App")
if page == "Upload":   
    st.title ("Data Upload")

    uploaded_file = st.file_uploader("Choose a File", type = ["csv"])

    if uploaded_file is not None:
        if st.button("Upload"):
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df
            st.success("File uploaded successfully")
            st.write("File Preview")
            st.dataframe(df)
    # if st.button ("Data Pre-processing"):
    #     st.switch_page()

elif page == "Pre-processing":
    st.title ("Data Pre-processing")

    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("Null Values Check")
        null_count = df.isnull().sum()
        st.write(null_count[null_count > 0])
        
        st.subheader("Fill Missing values")
        fill_columns = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                fill_option = st.selectbox(
                    f"How to fill '{col}' ? ",
                    ("None", "Mean", "Mode"),
                    key = f"fill_{col}"
                )
                fill_columns[col] = fill_option   

    if st.button("Apply Missing Value Filling"):
        for col, method in fill_columns.items():
            if method == "Fill with Mean" and df[col].dtype != "object":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Fill with Median" and df[col].dtype != "object":
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Fill with Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.session_state["df"] = df
        st.success("Missing values handled successfully!")

        st.subheader("Check Null Values (After Filling)")
        null_counts_after = df.isnull().sum()
        st.write(null_counts_after[null_counts_after > 0]
                 if null_counts_after.sum() > 0
                 else "No missing values left!")

    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("Convert Categorical columns To Numeric Columns")
        if st.checkbox("Apply Label Encoding"):
            if st.button("Convert"):
                le = LabelEncoder()
                for col in df.select_dtypes(include = "object").columns:
                    df[col] = le.fit_transform(df[col].astype(str))
                st.session_state["df"] = df
                st.success("Categorical columns converted successfully")
                st.write("File Preview")
                st.dataframe(df)

    if "df" in st.session_state:
        df = st.session_state["df"]

        st.subheader("Download Refined Data")
        csv = df.to_csv(index = False).encode("utf-8")
        st.download_button(
            label = "Download CSV",
            data = csv,
            file_name = "Refined_data.csv",
            mime = "text/csv"
        )

elif page == "Modeling":
    st.title ("Machine Learning")

    uploaded_file = st.file_uploader("Upload your refined CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(df)

        # target column
        target_col = st.selectbox("Select Target (y) Column", df.columns)

        if st.button("Run Models"):
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

            results = []
            models = {}

            # linear regress
            if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10: 
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)
                score = lin_reg.score(X_test, y_test)  
                results.append(("Linear Regression", score))
                models["Linear Regression"] = lin_reg

            # logistic regress
            if len(np.unique(y)) <= 10:  # Classification
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(X_train, y_train)
                y_pred = log_reg.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results.append(("Logistic Regression", acc))
                models["Logistic Regression"] = log_reg

                st.write("**Logistic Regression Confusion Matrix:**")
                st.write(confusion_matrix(y_test, y_pred))
                st.text("Classification Report:\n" + classification_report(y_test, y_pred))

            # randomforest
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            results.append(("Random Forest", acc_rf))
            models["Random Forest"] = rf

            st.write("**Random Forest Confusion Matrix:**")
            st.write(confusion_matrix(y_test, y_pred_rf))
            st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))

            # knn
            knn = KNeighborsClassifier()
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            acc_knn = accuracy_score(y_test, y_pred_knn)
            results.append(("KNN", acc_knn))
            models["KNN"] = knn

            st.write("**KNN Confusion Matrix:**")
            st.write(confusion_matrix(y_test, y_pred_knn))
            st.text("Classification Report:\n" + classification_report(y_test, y_pred_knn))

            # model comparison
            st.subheader("Model Comparison")
            results_df = pd.DataFrame(results, columns=["Model", "Score"])
            st.dataframe(results_df)

            # save model
            best_model_name, best_score = max(results, key=lambda x: x[1])
            best_model = models[best_model_name]

            with open("model.pkl", "wb") as f:
                pickle.dump(best_model, f)

            st.success(f"Best Model: {best_model_name} (Score: {best_score:.2f}) saved as model.pkl")

