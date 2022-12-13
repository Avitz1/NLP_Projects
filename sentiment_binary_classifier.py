from sys import argv
import numpy as np
import pandas as pd
import spacy
from spacy import tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    # Load data
    fields = ['Review Text', 'Recommended IND']
    df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', skipinitialspace=True, usecols=fields)
    df = df.dropna()
    tk = spacy.blank('en').tokenizer

    # Tokenize data
    df['Review Text'] = df.apply(lambda row: tk(row['Review Text']), axis=1)
    # # Remove stop words
    # df['Review Text'] = df['Review Text'].apply(lambda x: [token for token in x if not token.is_stop])
    # Remove whitespace
    df['Review Text'] = df['Review Text'].apply(lambda x: [token for token in x if not token.is_space])
    # Remove empty strings
    df['Review Text'] = df['Review Text'].apply(lambda x: [token for token in x if token.text != ''])
    # Transform the tokens back into text
    df['Review Text'] = df['Review Text'].apply(lambda x: ' '.join([token.text for token in x]))
    #make X to be the review text and y to be the recommended indicator
    X = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # find which is the majority class
    majority_class = np.bincount(y).argmax()

    # find indexes of the majority class
    majority_class_indexes = np.where(y == majority_class)[0]

    # downsample the majority class
    majority_class_indexes_downsampled = np.random.choice(majority_class_indexes,
                                size=len(X)-len(majority_class_indexes), replace=False)
    # combine the majority class downsampled with the minority class
    indexes = np.concatenate((majority_class_indexes_downsampled, np.where(y != majority_class)[0]))
    # shuffle the indexes
    np.random.shuffle(indexes)
    # downsample the data
    X = X[indexes]
    y = y[indexes]

    # # find best k
    # vectorizer = TfidfVectorizer()
    # X = vectorizer.fit_transform(X)
    # X = X.toarray()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # best_score = 0
    # best_k = 0
    # for k in range(1, 100, 3):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, X_train, y_train, cv=5)
    #     if scores.mean() > best_score:
    #         best_score = scores.mean()
    #         best_k = k
    # print(best_k)

    # Create feature vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    # X = X.toarray()

    # Create feature vectors using CountVectorizer
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(X)
    # X = X.toarray()

    # # Standardize the data
    # scaler = StandardScaler(with_mean=False)
    # X = scaler.fit_transform(X)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Perform classification with KNN
    knn = KNeighborsClassifier(n_neighbors=79)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    #print results
    print(classification_report(y_test, y_pred))

    # Perform 10-fold cross validation
    scores = cross_val_score(knn, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

