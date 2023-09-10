import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def preprocess_and_predict():
    df = pd.read_csv('loan_prediction.csv')

    df = df.drop('Loan_ID', axis=1)
    df.isnull().sum()


    # Fill missing values in categorical columns with mode
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

    # Fill missing values in LoanAmount with the median
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    # Fill missing values in Loan_Amount_Term with the mode
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

    # Fill missing values in Credit_History with the mode
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


    # Calculate the IQR
    Q1 = df['ApplicantIncome'].quantile(0.25)
    Q3 = df['ApplicantIncome'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df = df[(df['ApplicantIncome'] >= lower_bound) & (df['ApplicantIncome'] <= upper_bound)]

    # Calculate the IQR
    Q1 = df['CoapplicantIncome'].quantile(0.25)
    Q3 = df['CoapplicantIncome'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df = df[(df['CoapplicantIncome'] >= lower_bound) & (df['CoapplicantIncome'] <= upper_bound)]

    # Convert categorical columns to numerical using one-hot encoding
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    df = pd.get_dummies(df, columns=cat_cols)

    # Split the dataset into features (X) and target (y)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split the dataset into features (X) and target (y)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Scale the numerical columns using StandardScaler
    scaler = StandardScaler()
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    
    model = SVC(random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    # Convert X_test to a DataFrame
    X_test_df = pd.DataFrame(X_test, columns=X_test.columns)

    # Add the predicted values to X_test_df
    X_test_df['Loan_Status_Predicted'] = y_pred
    
    return X_test_df


if __name__=="__main__":
    
    result_df = preprocess_and_predict()
    print(result_df)