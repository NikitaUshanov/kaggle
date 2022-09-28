import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('/Users/nikitaushanov/Downloads/credit_train.csv')
# print(df)

df = df.drop(["Loan ID", "Customer ID"], axis=1)

def df_nulls():
    df.isnull().sum()

def text_to_int(data, type_df = 'train'):
    le = preprocessing.LabelEncoder()
    if type_df == 'train':
        columns = ["Years in current job", "Loan Status", "Term", "Home Ownership", "Purpose"]
    else:
        columns = ["Years in current job", "Term", "Home Ownership", "Purpose"]
    for col in columns:
        data[col] = le.fit_transform(data[col])
        print(le.classes_)

def clean(data):
    cols = ["Credit Score", "Annual Income", "Years in current job", "Months since last delinquent", "Bankruptcies", "Tax Liens", "Maximum Open Credit"]

    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)

    return data


text_to_int(df)
clean(df)
# print(df)
# df_nulls()