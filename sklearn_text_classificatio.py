
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pandas as pd

import argparse
import os

from sklearn import svm
from sklearn.externals import joblib

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file)
                   for file in os.listdir(args.train)]

    filepath_dict = {'docaitrust':   input_files[0]}

    df_list = []
    # for source, filepath in filepath_dict.items():
    #     df = pd.read_csv(filepath, names=['label', 'sentence'], sep=',')
    #     df['source'] = source  # Add another column filled with the source name
    #     df_list.append(df)

    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['label', 'sentence'], sep=',')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)
    print(df.iloc[0])

    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    # print(vectorizer.vocabulary_)

    df_docaitrust = df[df['source'] == 'docaitrust']
    sentences = df_docaitrust['sentence'].values
    y = df_docaitrust['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    # print(vectorizer.vocabulary_)

    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    # print(X_train.toarray())

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    # joblib.dump(classifier,  "sklearn_text_classifier.joblib")

    # output_channel = prefix + '/output'
    # validation_channel = prefix + '/validation'

    # sess.upload_data(path='sklearn_text_classifier.joblib',
    #                  bucket=bucket, key_prefix=output_channel)

    print("Accuracy:", score)

    result = classifier.predict(vectorizer.transform(
        ["Date of this deed settlement is 9th Jan 2012 "]))
    print(result)

    joblib.dump(classifier, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
