import pandas as pd
import texthero as th
from texthero import preprocessing

train_data = pd.read_csv('./data/train.csv')
dev_data = pd.read_csv('./data/dev.csv')

# https://texthero.org/docs/getting-started#preprocessing
preprocess_pipeline = [preprocessing.fillna,
                       preprocessing.lowercase,
                       preprocessing.remove_whitespace]

train_data['text'] = train_data['text'].pipe(th.clean, preprocess_pipeline)
# Convert bool to int: True = 1, False = 0
train_data['humor'] = train_data['humor'].astype(int)

dev_data['text'] = dev_data['text'].pipe(th.clean, preprocess_pipeline)
dev_data['humor'] = dev_data['humor'].astype(int)

# Save preprocessed data
train_data.to_csv('./data/train_clean.csv', index = False)
dev_data.to_csv('./data/dev_clean.csv', index = False)

###### For text generation ######
data = pd.read_csv('./data/dataset.csv')

data['humor'] = data['humor'].astype(int)

# Make a copy of the original dataframe
humor_only = data.copy()

# Select rows that are labelled as humorous
humor_only = humor_only.loc[humor_only['humor'] == 1]
humor_only.drop(['humor'], axis = 1, inplace = True)

humor_only.to_csv('./data/humor_text_only.csv', index = False)