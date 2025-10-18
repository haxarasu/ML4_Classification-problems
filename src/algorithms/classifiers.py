import numpy as np

class s21LogisticRegression:
    def __init__(self, lr=0.1, epochs=50, batch_size=128, random_state=42, threshold=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.threshold = threshold
        self.w = None
        self.b = 0.0


    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40, 40)

        return 1.0 / (1.0 + np.exp(-z))


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0.0, 0.01, size=d)
        self.b_ = 0.0

        for _ in range(self.epochs):
            idx = rng.permutation(n)
            Xs, ys = X[idx], y[idx]

            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                Xb = Xs[start:stop]
                yb = ys[start:stop]
                m = len(Xb)

                z = Xb @ self.w_ + self.b_
                p = self._sigmoid(z)

                grad_w = (Xb.T @ (p - yb)) / m
                grad_b = np.mean(p - yb)

                self.w_ -= self.lr * grad_w
                self.b_ -= self.lr * grad_b

        return self


    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = self._sigmoid(X @ self.w_ + self.b_)

        return np.vstack([1.0 - p1, p1]).T


    def predict(self, X):
        thr = float(0.5) if self.threshold is None else float(self.threshold)
        p1 = self.predict_proba(X)[:, 1]

        return (p1 >= 0.5).astype(int)


class s21KNN:
    def __init__(self, weights="uniform", n_neighbors=5, threshold=None):
        assert weights in ("uniform", "distance")
        self.k = n_neighbors
        self.weights = weights
        self.threshold = threshold
        self.X_ = None
        self.y_ = None
        self.classes_ = None

    @staticmethod
    def _euclidean(A, B):
        A2 = np.sum(A**2, axis=1, keepdims=True)
        B2 = np.sum(B**2, axis=1, keepdims=True).T
        D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)

        return np.sqrt(D2)


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")    

        self.X_ = X
        self.y_ = y

        self.classes_ = np.unique(y)

        return self


    def predict_proba(self, X):
        if self.X_ is None:
            raise RuntimeError("Call fit method before calling predict_proba method")
        X = np.asarray(X, dtype=float)

        D = self._euclidean(X, self.X_)

        # search for k indexes of nearest neighbors
        k = min(self.k, self.X_.shape[0])
        nn_idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]

        if self.weights == "uniform":
            W = np.ones_like(nn_idx, dtype=float)
        else: # if self.weights == "distance": distance contributes on the weights" coefs
            rows = np.arange(X.shape[0])[:, None]
            dists = D[rows, nn_idx]
            W = 1.0 / (dists + 1e-12)

        n_test = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_test, n_classes), dtype=float)

        neigh_labels = self.y_[nn_idx]
        for ci, cls in enumerate(self.classes_):
            votes = (neigh_labels == cls).astype(float) # bool matrix -> 1.0/0.0 matrix
            proba[:, ci] = np.sum(W * votes, axis=1)    # marking neighbours" weights only if cls marked

        s = proba.sum(axis=1, keepdims=True)
        np.divide(proba, s, out=proba, where=s > 0)

        return proba


    def predict(self, X):
        proba = self.predict_proba(X)
        
        if len(self.classes_) == 2:
            thr = float(0.5) if self.threshold is None else float(self.threshold)
            if 1 in self.classes_: # if marks are {0,1} -> finding index of class 1
                pos_idx = int(np.where(self.classes_ == 1)[0][0])
                return (proba[:, pos_idx] >= thr).astype(int)
            else: # if marks are not {0,1} -> using argmax()
                return self.classes_[np.argmax(proba, axis=1)]
        else: # multiclass -> choosing a class with max proba
            return self.classes_[np.argmax(proba, axis=1)]


class s21NaiveBayes:
    def __init__(self, var_smoothing=1e-9, threshold=None):
        self.var_smoothing = float(var_smoothing)
        self.threshold = threshold
        self.classes_ = None
        self.theta_ = None          # mean by classes
        self.var_ = None            # dispersion by classes
        self.class_log_prior_ = None

    
    def _joint_log_likelihood(self, X): # returns sum of log probas
        X = np.asarray(X, dtype=float)
        # log N(x | mu, var) = -0.5 * [ log(2πvar) + (x - mu)^2 / var ]
        jll = []
        for i in range(len(self.classes_)):
            mu  = self.theta_[i]
            var = self.var_[i]

            log_prob = -0.5 * (np.log(2.0 * np.pi * var) + ((X - mu) ** 2) / var).sum(axis=1)
            jll.append(self.class_log_prior_[i] + log_prob)

        return np.vstack(jll).T  # shape: (n_samples, n_classes)


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features), dtype=float)
        self.var_   = np.zeros((n_classes, n_features), dtype=float)
        class_count = np.zeros(n_classes, dtype=float)

        for i, cls in enumerate(self.classes_):
            Xi = X[y == cls]
            if Xi.size == 0:
                self.theta_[i] = 0.0
                self.var_[i] = 1.0
                class_count[i] = 0.0
            else:
                class_count[i] = Xi.shape[0]
                self.theta_[i] = Xi.mean(axis=0)
                self.var_[i]   = Xi.var(axis=0) + self.var_smoothing

        total = class_count.sum()
        if total == 0:
            raise ValueError("Empty training data passed to Naive Bayes.")
        self.class_log_prior_ = np.log(class_count / total)
        
        return self


    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)

        a = jll - jll.max(axis=1, keepdims=True)
        P = np.exp(a)
        P /= P.sum(axis=1, keepdims=True)
        
        return P


    def predict(self, X): # same logics as in KNN .predict()
        proba = self.predict_proba(X)

        if len(self.classes_) == 2:
            thr = float(0.5) if self.threshold is None else float(self.threshold)
            if 1 in self.classes_:
                pos_idx = int(np.where(self.classes_ == 1)[0][0])
                return (proba[:, pos_idx] >= thr).astype(int)
            else:
                return self.classes_[np.argmax(proba, axis=1)]
        else:
            return self.classes_[np.argmax(proba, axis=1)]