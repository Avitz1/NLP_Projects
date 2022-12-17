import sys
from sys import argv
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
def tokenize(text, tk = spacy.blank('en').tokenizer):
    return tk(text)
def load_data_and_tokenize(path):
    fields = ['Review Text', 'Recommended IND']
    df = pd.read_csv(path, skipinitialspace=True, usecols=fields)
    # drop rows with empty review text
    df = df.dropna()
    return df
def balance_data(df):
    X = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    # find which is the majority class
    majority_class = np.bincount(y).argmax()
    # find indexes of the majority class
    majority_class_indexes = np.where(y == majority_class)[0]
    # downsample the majority class
    majority_class_indexes_downsampled = np.random.choice(majority_class_indexes,
                                        size=len(X) - len(majority_class_indexes), replace=False)
    # combine the majority class downsampled with the minority class
    indexes = np.concatenate((majority_class_indexes_downsampled, np.where(y != majority_class)[0]))
    # shuffle the indexes
    np.random.shuffle(indexes)
    print("Before Down-sampling:")
    print("Recommended: " + str(len(majority_class_indexes)))
    print("Not Recommended: " + str(len(majority_class_indexes_downsampled)) + "\n")
    print("After Down-sampling:")
    print("Recommended: " + str(len(majority_class_indexes_downsampled)))
    print("Not Recommended: " + str(len(majority_class_indexes_downsampled)) + "\n")
    return df.iloc[indexes, :], X[indexes], y[indexes]
def vectorize_using_BoW(X):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    return X
def build_specially_crafted_features(df):
    # create an empty feature vector of length 10 for each review
    X = np.zeros((len(df), 2))
    # tokenize
    df['Review Text'] = df['Review Text'].apply(tokenize)
    # remove empty tokens
    df['Review Text'] = df['Review Text'].apply(lambda x: [token for token in x if token.text != ''])
    positive_words = ["wonderful", "silky", "sexy", "comfortable", "love", "sooo", "pretty", "glad",
                      "nicely", "fun", "flirty", "fabulous", "compliments", "great", "flattering",
                      "perfect", "gorgeous", "perfectly", "feminine", "style", "such", "happy", "cute",
                      "cozy", "stylish", "classic", "beautifully", "super", "lovely", "unique", "roomy"
                       "adorable", "soft",  "loved"]
    # count positive words
    X[:, 0] = df['Review Text'].apply(lambda x: len([token for token in x if token.text.lower() in positive_words]))
    negative_words = ["outrageously", "ok", "back", "returned", "return", "runs", "running", "but", "disappointing",
                      "disappointed", "replace", "annoying", "not", "returning", "cheap", "poor", "terrible",
                      "unflattering", "funny", "cut", "unfortunately", "scratchy", "odd",
                      "uncomfortable", "stiff", "itchy", "torn"]

    # count negative words
    X[:, 1] = df['Review Text'].apply(lambda x: len([token for token in x if token.text.lower() in negative_words]))

    # X[:, 2] = df['Review Text'].apply(lambda x: len([token for token in x if token.is_stop]))
    # X[:, 3] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'VERB']))
    # X[:, 4] = df['Review Text'].apply(lambda x: len([token for token in x if token.is_punct]))
    # X[:, 5] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'ADV']))
    # X[:, 6] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'PRON']))
    # X[:, 7] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'NOUN']))
    # X[:, 8] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'PROPN']))
    # X[:, 9] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'CCONJ']))
    # X[:, 10] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'INTJ']))
    # X[:, 11] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'DET']))
    # X[:, 12] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'PART']))
    # X[:, 13] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'SYM']))
    # X[:, 14] = df['Review Text'].apply(lambda x: len([token for token in x if token.pos_ == 'ADJ']))
    # X[:, 15] = df['Review Text'].apply(lambda x: len([token for token in x if token.is_digit]))
    # # count the number of words in the review
    # X[:, 16] = df['Review Text'].apply(lambda x: len(x))
    # standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
# def find_best_k(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#     best_score = 0
#     best_k = 0
#     for k in range(1, 100, 3):
#         knn = KNeighborsClassifier(n_neighbors=k)
#         scores = cross_val_score(knn, X_train, y_train, cv=5)
#         if scores.mean() > best_score:
#             best_score = scores.mean()
#             best_k = k
#     return best_k
def split_train_test_print(knn, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    knn.fit(X_train, y_train)
    scores = cross_val_score(knn, X, y, cv=10)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(classification_report(y_test, knn.predict(X_test)))

if __name__ == "__main__":
    csv_file = argv[1]  # The path to thecsv file as downloaded from kaggle after unzipping
    output_file = argv[2]  # The path to the text file in which the output is written onto
    # csv_file = "Womens Clothing E-Commerce Reviews.csv"
    df = load_data_and_tokenize(csv_file)
    # balance the classes randomly and return a new dataframe in which the majority class is down-sampled
    with open(output_file, 'w', encoding='utf8') as f:
        sys.stdout = f
        df, X, y = balance_data(df)

        X_BoW = vectorize_using_BoW(X)
        X_custom_features = build_specially_crafted_features(df)
        # initiaze the knn classifier, the following line can be used to find a good value for k
        # best_k = find_best_k(X, y)
        # 59 was found to be a decent k value for this dataset with BoW
        knn = KNeighborsClassifier(59)

        print("== BoW Classification ==")
        split_train_test_print(knn, X_BoW, y)

        print("== Custom Feature Vector Classification ==")
        # 19 gave already a good result for this dataset with custom features
        knn = KNeighborsClassifier(19)
        split_train_test_print(knn, X_custom_features, y)
        sys.stdout = sys.__stdout__
        f.close()