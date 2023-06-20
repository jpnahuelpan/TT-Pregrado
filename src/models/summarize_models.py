import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords


class Summarize:
    """Clase que contiene los metodos para resumir los grupos o cluster
        de microtextos."""
    import numpy as np

    def LSA_summary(A, K):
        # Compute the SVD of the matrix A
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        # Create the diagonal matrix Sigma
        Sigma = np.diag(s)
        V = VT.T
        # Compute the sentence scores
        sentence_scores = np.linalg.norm(V @ Sigma, axis=1)

        # Get the indices of the sentences with the highest K scores
        summary_indices = np.argsort(sentence_scores)[::-1][:K]

        return summary_indices

    def TCRE(corpus, cluster_result, vectorizer):
        # Step 1: Set variables
        n = len(corpus)
        feat_list = []
        # vectorizer = CountVectorizer()
        vectorizer.fit(corpus)
        # Step 2: Map text to 0-1 bag-of-words features
        for i in range(n):
            # Step 5: Split text into tokens
            tokens = corpus[i].split()

            # Step 6: Filter out stop words and low-frequency words
            stop_words = stopwords.words('spanish')
            tokens = [token for token in tokens if token not in stop_words]
            word_counts = {word: tokens.count(word) for word in set(tokens)}
            word_counts = [k for k, v in word_counts.items() if v >= 2]

            # Step 7: Map tokens into 0-1 feature vector feat
            feat = np.zeros(len(vectorizer.vocabulary_))
            for word, index in vectorizer.vocabulary_.items():
                if word in word_counts:
                    feat[index] = 1

            # Step 8: Append feature vector feat into feat_list
            feat_list.append(feat)

        # Step 11: Train logistic regression classifier
        lg_classifier = LogisticRegression(
            random_state=0, max_iter=10000).fit(feat_list, cluster_result)

        # Step 13: Get weight list
        weight_list = lg_classifier.coef_

        # Step 15-24: Obtain Indication Words for Every Cluster
        ind_words_list = []
        for weight in weight_list:
            tmp_weight = []
            for w in weight:
                tmp_weight.append(abs(w))
            tmp_idx = np.argsort(-np.array(tmp_weight), kind='quicksort')[:n]
            ind_words = []
            for idx in tmp_idx:
                ind_words.append(
                    list(vectorizer.vocabulary_.keys())[list(
                        vectorizer.vocabulary_.values()).index(idx)])
            ind_words_list.append(ind_words)

        return ind_words_list
