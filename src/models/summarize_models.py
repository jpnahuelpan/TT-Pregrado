import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords


class Summarize:
    """Clase que contiene los metodos para resumir los grupos o cluster
        de microtextos."""

    def LSA_summary(A, K):
        # Calcular SVD de la matriz A
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        # Se crea la matriz diagonal Sigma
        Sigma = np.diag(s)
        V = VT.T
        # Calculos de la puntuaciones de la oraciones
        sentence_scores = np.linalg.norm(V @ Sigma, axis=1)

        # Se obtiene las K puntuaciones más altas
        summary_indices = np.argsort(sentence_scores)[::-1][:K]

        return summary_indices

    def log_regresion_weights(self, X, y):
        # para garantizar la reproducibilidad de los resultados.
        np.random.seed(5)
        torch.manual_seed(5)

        num_features = len(X[0])
        num_classes = len(set(y))
        input_data = torch.tensor(np.array(X))
        target = torch.tensor(np.array(y))

        # Definición del modelo de regresión logística
        class LogisticRegression(nn.Module):
            def __init__(self, num_features, num_classes):
                super(LogisticRegression, self).__init__()
                self.weights = nn.Parameter(torch.randn(
                    num_classes, num_features))
                self.num_clases = num_classes

            # Implementación de la función de aprendizaje TCRE
            def tcre_learning_fn(self, x):
                XW = torch.matmul(x, self.weights.t())
                probabilities = torch.zeros(x.size(0), self.num_clases)
                # Calc: p(yi=l|xi)= exp(wl x xi) / sum(exp(wj x xi))
                # where: j=1, 2, ..C; C=num_clases
                for i in range(x.size(0)):
                    for l in range(self.num_clases):
                        probabilities[i][l] = torch.exp(XW[i][l]) / torch.sum(torch.exp(XW[i][:]))
                return probabilities

            def forward(self, x):
                probabilities = self.tcre_learning_fn(x)
                return probabilities

        model = LogisticRegression(num_features, num_classes)

        # Definir la función de pérdida y el optimizador
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # Step 11: Train logistic regression classifier
        # Realizar el entrenamiento
        for _ in range(10000):
            output = model(input_data.float())
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Obtener los pesos del modelo
        weights = model.weights

        return weights.tolist()

    def TCRE(self, corpus, cluster_result, vectorizer):
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

        # Step 13: Get weight list
        weight_list = self.log_regresion_weights(feat_list, cluster_result)

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
