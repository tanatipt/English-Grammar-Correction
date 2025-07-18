import pandas as pd
from unidecode import unidecode
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

def preprocess_df(df : pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by cleaning text columns and removing unwanted characters.

    Args:
        df (pd.DataFrame): DataFrame containing the columns 'modified' and 'sentence'.
    """
    
    def preprocess_text(column : pd.Series) -> pd.Series:
        """
        Preprocess a text column by stripping whitespace, replacing multiple spaces with a single space,

        Args:
            column (pd.Series): Text column to preprocess.

        Returns:
            pd.Series: Preprocessed text column with unwanted characters removed.
        """
        # Strip whitespace, replace multiple spaces with a single space, and remove unwanted characters
        strip_col = column.str.strip()
        wsr_col = strip_col.str.replace(r"\s+", " ", regex=True)
        bad_punc_col = wsr_col.str.replace(r'"#%&\*\+/<=>@[\\]^{|}~_', '', regex=True)
        return bad_punc_col.apply(unidecode)
    
    df['modified'] = preprocess_text(df['modified'].str.replace(f"fix grammar:", ""))
    df['sentence'] = preprocess_text(df['sentence'])

    return df

# Load the data
train_data = pd.read_parquet("source/train.parquet", engine='pyarrow')
test_data  = pd.read_parquet("source/test.parquet",  engine='pyarrow')
data = pd.concat([train_data, test_data]).drop(columns=['__index_level_0__', 'sec_transformation'])

# Filter for English language and preprocess
data = data[data['lang'] == 'en'].drop(columns="lang")
preprocessed_data = preprocess_df(data).dropna()

# Split the data into train, validation, test, and early stopping datasets
X_train, X_test = train_test_split(preprocessed_data,test_size=2000, random_state=2543673, stratify=preprocessed_data['transformation'] )
X_train, X_valid = train_test_split(X_train,  test_size=2000, random_state=2543673, stratify=X_train['transformation'])
X_train, X_es = train_test_split(X_train,train_size=2000, test_size=2000, random_state=2543673, stratify=X_train['transformation'])

# Save the datasets to CSV files
X_train.to_csv("preprocessed/train.csv", index=False)
X_valid.to_csv("preprocessed/valid.csv", index=False)
X_test.to_csv("preprocessed/test.csv", index=False)
X_es.to_csv('preprocessed/es.csv', index = False)
            
            