import pandas as pd
from unidecode import unidecode
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

def preprocess_df(df):
    
    def preprocess_text(column):
        strip_col = column.str.strip()
        wsr_col = strip_col.str.replace(r"\s+", " ", regex=True)
        bad_punc_col = wsr_col.str.replace(r'"#%&\*\+/<=>@[\\]^{|}~_', '', regex=True)
        return bad_punc_col.apply(unidecode)
    
    df['modified'] = preprocess_text(df['modified'].str.replace(f"fix grammar:", ""))
    df['sentence'] = preprocess_text(df['sentence'])

    return df

train_data = pd.read_parquet("source/train.parquet", engine='pyarrow')
test_data  = pd.read_parquet("source/test.parquet",  engine='pyarrow')
data = pd.concat([train_data, test_data]).drop(columns=['__index_level_0__', 'sec_transformation'])
data = data[data['lang'] == 'en'].drop(columns="lang")
preprocessed_data = preprocess_df(data).dropna()

X_train, X_test = train_test_split(preprocessed_data,test_size=0.1, random_state=2543673, stratify=preprocessed_data['transformation'] )
X_train, X_valid = train_test_split(X_train,  test_size=0.1, random_state=2543673, stratify=X_train['transformation'])

X_train.to_csv("preprocessed/train.csv", index=False)
X_valid.to_csv("preprocessed/valid.csv", index=False)
X_test.to_csv("preprocessed/test.csv", index=False)
            
            