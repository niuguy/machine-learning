import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn import *
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import nltk


def text_tfidf(df):
    count_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,
                                       preprocessor=None, stop_words='english', max_features=None)
    tfidf = count_vectorizer.fit_transform(df['Text'])

    svd = TruncatedSVD(n_components=50, n_iter=25, random_state=12)
    truncated_tfidf = svd.fit_transform(tfidf)
    return truncated_tfidf


def do_vectorization(raw_df, key):
    count_vectorizer = feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 8))
    vectorized_df = count_vectorizer.fit_transform(raw_df[key])
    svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
    truncated_df = svd.fit_transform(vectorized_df)
    return truncated_df


def extend_gene_variant_features(raw_df):
    # count the appearance of gene
    df = pd.DataFrame()
    df['Gene_Share'] = raw_df.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]),
                                axis=1)
    df['Variation_Share'] = raw_df.apply(
        lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

    for c in raw_df.columns:
        if raw_df[c].dtype == 'object':
            if c in ['Gene', 'Variation']:
                lbl = LabelEncoder()
                df[c + '_lbl_enc'] = lbl.fit_transform(raw_df[c].values)
                df[c + '_len'] = raw_df[c].map(lambda x: len(str(x)))
                df[c + '_words'] = raw_df[c].map(lambda x: len(str(x).split(' ')))
            elif c != 'Text':
                lbl = LabelEncoder()
                df[c] = lbl.fit_transform(raw_df[c].values)
            if c == 'Text':
                df[c + '_len'] = raw_df[c].map(lambda x: len(str(x)))
                df[c + '_words'] = raw_df[c].map(lambda x: len(str(x).split(' ')))

    gene_vectorized = do_vectorization(raw_df, 'Gene')
    variation_vectorized = do_vectorization(raw_df, 'Variation')

    return np.hstack((df.values, gene_vectorized, variation_vectorized))


def lr_fit(train_x, train_y):

    # probas = cross_val_predict(LogisticRegression(), train_x, train_y, cv=StratifiedKFold(random_state=8),
    #                           n_jobs=-1, method='predict_proba', verbose=2)

    clf = SVC()
    clf.fit(train_x, train_y)
    probas = clf.predict_proba(train_x)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(train_y)
    preds = classes[pred_indices]
    print('LR Accuracy: {}'.format(accuracy_score(train_y, preds)))
    print('LR Log loss: {}'.format(log_loss(train_y, probas)))

def xgb_cv_fit(train_X, train_y):

    cls = xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, random_state=0)

    param_grid = {'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.3], 'n_estimators': [25, 50], 'gamma': [0, 0.1]}
    model = grid_search.GridSearchCV(estimator=cls, param_grid=param_grid, scoring='accuracy', verbose=10,
                                     n_jobs=1, iid=True, refit=True, cv=5)

    model.fit(train_X, train_y)

    probas = model.predict_proba(train_X)
    print("xgb best accuracy : %0.3f" % model.best_score_)
    print("xgb best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print('xgb log loss: {}'.format(log_loss(train_y, probas)))


def run_xgb(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.03333
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 9
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        model = xgb.train(plst, xgtrain, num_rounds)

    return model


def main():

    # training and test files, created using SRK's python script
    train_file = "train_stacknet.csv"
    test_file = "test_stacknet.csv"

    # read files
    train_variants_df = pd.read_csv("input/training_variants")
    # test_variants_df = pd.read_csv("input/test_variants")[:100]
    train_text_df = pd.read_csv("input/training_text", sep="\|\|", engine='python', header=None, skiprows=1,
                                names=["ID", "Text"])
    # test_text_df = pd.read_csv("input/test_text", sep="\|\|", engine='python', header=None, skiprows=1,
    #                            names=["ID", "Text"])[:10]

    # merge
    train_df = pd.merge(train_variants_df, train_text_df, how='left', on='ID').fillna('')
    # test_df = pd.merge(test_variants_df, test_text_df, how='left', on='ID').fillna('')
    train_y = np.array(train_df['Class'].apply(lambda x: x - 1))



    # extend features
    tr_variants_le = extend_gene_variant_features(train_df)
    # te_variants_le = extend_gene_variant_features(test_df)

    # Text-tfidf
    tr_text_tfidf = text_tfidf(train_df)
    # te_text_tfidf = text_tfidf(test_df)

    # merge all features
    train_x =np.hstack((tr_variants_le, tr_text_tfidf))
    # test_x = np.hstack((te_variants_le, te_text_tfidf))

    # print "train_x"

    # logistic benchmark
    lr_fit(train_x, train_y)
    # xgb boost
    xgb_cv_fit(train_x, train_y)

    print 'Training DONE'


if __name__ == '__main__':

    main()
