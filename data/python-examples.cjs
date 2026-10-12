// Самостоятельные учебные мини-примеры, не цитаты из книг.
// Обычный Python 3; матричные примеры используют NumPy.
module.exports = {
  'unit-step-function': `z = -0.4
prediction = 1 if z >= 0 else -1  # сравниваем сумму с порогом
print(prediction)`,
  'net-input': `weights = [-1, 0.5, 1]
x = [1, 2, 3]  # первая единица отвечает за свободный член
z = sum(w * value for w, value in zip(weights, x))
print(z)`,
  'perceptron-rule': `eta, y, prediction, x, weight = 0.1, 1, -1, 2, 0.5
delta = eta * (y - prediction) * x
print(delta, weight + delta)`,
  'sse-cost': `y, prediction = [1, 3], [2, 2]
loss = sum((a - b)**2 for a, b in zip(y, prediction)) / 2
print(loss)`,
  'batch-gradient-descent': `weight, gradient, eta = 4, 3, 0.1
delta = -eta * gradient  # идём против роста ошибки
print(delta, weight + delta)`,
  'sse-gradient': `y, prediction, x = 3, 2, 4
gradient = -(y - prediction) * x
print(gradient)`,
  'standardization': `x, mean, std = 185, 170, 10
print((x - mean) / std)  # расстояние от среднего в единицах std`,
  'sgd-update': `eta, y, prediction, x = 0.1, 1, 0.6, 2
delta = eta * (y - prediction) * x
print(round(delta, 6))`,
  'adaptive-learning-rate': `c1, c2 = 1, 10
for iteration in [0, 90]:
    print(iteration, c1 / (iteration + c2))`,
  'odds-ratio': `p = 0.75
print(p / (1 - p))`,
  'logit': `from math import log
p = 0.8
print(round(log(p / (1 - p)), 6))`,
  'sigmoid': `from math import exp
z = 2
p = 1 / (1 + exp(-z))
print(round(p, 6))`,
  'likelihood': `y, p = [1, 0], [0.8, 0.3]
likelihood = 1.0
for label, probability in zip(y, p):
    likelihood *= probability**label * (1 - probability)**(1 - label)
print(round(likelihood, 6))`,
  'log-likelihood': `from math import log
y, p = [1, 0], [0.8, 0.3]
terms = [label * log(prob) + (1 - label) * log(1 - prob)
         for label, prob in zip(y, p)]
print([round(term, 6) for term in terms])
print(round(sum(terms), 6))`,
  'logistic-cost': `from math import log
y, p = [1, 0], [0.8, 0.3]
loss = -sum(label * log(prob) + (1 - label) * log(1 - prob)
            for label, prob in zip(y, p))
print(round(loss, 6))  # сумма; для среднего разделите на len(y)`,
  'sigmoid-derivative': `p = 0.8
derivative = p * (1 - p)
print(round(derivative, 6))`,
  'logistic-regression-weight-update': `weight, eta = 0.5, 0.1
y, p, x = [1, 0], [0.8, 0.3], [2, 1]
delta = eta * sum((label - prob) * feature
                  for label, prob, feature in zip(y, p, x))
print(round(delta, 6), round(weight + delta, 6))`,
  'log-likelihood-partial-weight': `y, p, x = 1, 0.8, 2
dp_dw = p * (1 - p) * x
dl_dw = (y / p - (1 - y) / (1 - p)) * dp_dw
print(round(dl_dw, 6))`,
  'sigmoid-chain-weight': `p, x = 0.8, 2
dp_dz = p * (1 - p)
dz_dw = x
print(round(dp_dz * dz_dw, 6))`,
  'logistic-gradient-component': `y, p, x = [1, 0], [0.8, 0.3], [2, 1]
gradient = -sum((label - prob) * feature
                for label, prob, feature in zip(y, p, x))
print(round(gradient, 6))  # градиент отрицательного log-likelihood`,
  'l2-regularization': `weights, lam = [3, 4], 0.2
penalty = lam / 2 * sum(w*w for w in weights)
print(penalty, 1 / lam)  # штраф и C для этой записи`,
  'svm-margin': `from math import sqrt
weights = [3, 4]
print(2 / sqrt(sum(w*w for w in weights)))`,
  'svm-soft-margin': `weights, slack, C = [1, 2], [0, 0.5, 1], 2
objective = sum(w*w for w in weights) / 2 + C * sum(slack)
print(objective)`,
  'kernel-trick': `x, y = [1, 2], [2, 1]
phi = lambda v: [v[0]**2, 2**0.5 * v[0]*v[1], v[1]**2]
dot = lambda a, b: sum(x*y for x, y in zip(a, b))
print(round(dot(phi(x), phi(y)), 6), dot(x, y)**2)`,
  'rbf-kernel': `from math import exp
x, y, gamma = [1, 2], [2, 4], 0.5
distance_squared = sum((a-b)**2 for a, b in zip(x, y))
print(round(exp(-gamma * distance_squared), 6))`,
  'information-gain': `parent_impurity = 0.5
children = [(6, 0.25), (4, 0.0)]  # размер и неоднородность
gain = parent_impurity - sum(size / 10 * impurity for size, impurity in children)
print(round(gain, 6))`,
  'entropy': `from math import log2
probabilities = [0.5, 0.5]
entropy = -sum(p * log2(p) for p in probabilities if p > 0)
print(entropy)  # биты, поскольку логарифм по основанию 2`,
  'gini-impurity': `probabilities = [0.5, 0.5]
print(1 - sum(p*p for p in probabilities))`,
  'classification-error': `probabilities = [0.7, 0.3]
print(round(1 - max(probabilities), 6))`,
  'minkowski-distance': `x, y = [1, 2], [4, 6]
for p in [1, 2]:
    distance = sum(abs(a-b)**p for a, b in zip(x, y))**(1/p)
    print(p, distance)`,
  'min-max-scaling': `x, minimum, maximum = 15, 10, 20
print((x - minimum) / (maximum - minimum))`,
  'l1-regularization': `weights = [-2, 3]
print(sum(abs(w) for w in weights))`,
  'covariance': `import numpy as np
x, y = np.array([1., 2., 3.]), np.array([2., 4., 6.])
cov = ((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1)
print(cov)  # выборочная ковариация: знаменатель n-1`,
  'eigen-decomposition': `import numpy as np
matrix = np.array([[2., 0.], [0., 1.]])
values, vectors = np.linalg.eigh(matrix)
print(values)
print(np.allclose(matrix @ vectors, vectors * values))`,
  'explained-variance-ratio': `eigenvalues = [4, 1]
print([value / sum(eigenvalues) for value in eigenvalues])`,
  'pca-projection': `import numpy as np
x = np.array([2., 3.])  # уже центрированные признаки
W = np.array([[1.], [0.]])
print(x @ W)  # из двух координат в одну`,
  'lda-mean-vector': `import numpy as np
class_samples = np.array([[1., 2.], [3., 4.]])
print(class_samples.mean(axis=0))`,
  'lda-within-scatter': `import numpy as np
classes = [np.array([[1., 2.], [3., 4.]]), np.array([[6., 3.], [8., 5.]])]
scatter = np.zeros((2, 2))
for samples in classes:
    centered = samples - samples.mean(axis=0)
    scatter += centered.T @ centered
print(scatter)`,
  'lda-between-scatter': `import numpy as np
classes = [np.array([[1., 2.], [3., 4.]]), np.array([[6., 3.], [8., 5.]])]
overall = np.concatenate(classes).mean(axis=0)
scatter = np.zeros((2, 2))
for samples in classes:
    delta = samples.mean(axis=0) - overall
    scatter += len(samples) * np.outer(delta, delta)
print(scatter)`,
  'kernel-matrix': `import numpy as np
X = np.array([[1., 2.], [3., 4.]])
print(X @ X.T)  # линейное ядро: phi(X) = X`,
  'polynomial-kernel': `x, y, degree, offset = [1, 2], [2, 1], 2, 1
print((sum(a*b for a, b in zip(x, y)) + offset)**degree)`,
  'sigmoid-kernel': `from math import tanh
x, y, gamma, offset = [1, 2], [2, 1], 0.5, 0
print(round(tanh(gamma * sum(a*b for a, b in zip(x, y)) + offset), 6))`,
  'kernel-centering': `import numpy as np
K = np.array([[1., 2.], [2., 4.]])
H = np.eye(len(K)) - np.ones_like(K) / len(K)
centered = H @ K @ H
print(centered)
print(centered.mean(axis=0))`,
  'error-accuracy': `truth, predicted = [1, 0, 1, 1], [1, 0, 0, 1]
error = sum(a != b for a, b in zip(truth, predicted)) / len(truth)
print(error, 1 - error)`,
  'fpr-tpr': `FP, TN, TP, FN = 4, 16, 8, 2
print(FP / (FP + TN), TP / (TP + FN))`,
  'precision-recall': `TP, FP, FN = 8, 2, 4
print(TP / (TP + FP), round(TP / (TP + FN), 6))`,
  'f1-score': `precision, recall = 0.8, 0.5
print(round(2 * precision * recall / (precision + recall), 6))`,
  'majority-voting': `from collections import Counter
votes = ['cat', 'dog', 'cat']
print(Counter(votes).most_common(1)[0][0])`,
  'ensemble-error': `from math import comb
n, error = 3, 0.2
probability = sum(comb(n, k) * error**k * (1-error)**(n-k)
                  for k in range(n//2 + 1, n+1))
print(round(probability, 6))  # предполагаем независимые ошибки`,
  'weighted-majority-voting': `votes, weights = [1, 0], [0.7, 0.3]
scores = {label: sum(w for vote, w in zip(votes, weights) if vote == label)
          for label in set(votes)}
print(max(scores, key=scores.get))`,
  'probability-voting': `probabilities = [[0.1, 0.9], [0.4, 0.6]]
scores = [sum(row[i] for row in probabilities) for i in range(2)]
print(scores, max(range(2), key=lambda i: scores[i]))`,
  'adaboost': `from math import log, exp
error = 0.2
alpha = 0.5 * log((1-error) / error)
weights, y, predicted = [0.5, 0.5], [1, -1], [1, 1]
updated = [w * exp(-alpha * target * guess)
           for w, target, guess in zip(weights, y, predicted)]
print(round(alpha, 6))
print([round(w / sum(updated), 6) for w in updated])`,
  'tf-idf': `from math import log
tf, n_documents, document_frequency = 3, 100, 9
idf = log(n_documents / (1 + document_frequency))  # вариант записи в конспекте
print(round(idf, 6), round(tf * idf, 6))`,
  'multiple-linear-regression': `intercept, weights, x = 20, [2, -1], [50, 3]
print(intercept + sum(w*value for w, value in zip(weights, x)))`,
  'pearson-correlation': `import numpy as np
x, y = [1, 2, 3], [2, 4, 6]
print(round(float(np.corrcoef(x, y)[0, 1]), 6))`,
  'mse': `truth, prediction = [2, 4], [3, 2]
print(sum((y-p)**2 for y, p in zip(truth, prediction)) / len(truth))`,
  'r-squared': `residual_sum, total_sum = 20, 100
print(1 - residual_sum / total_sum)`,
  'ridge-regression': `loss, weights, lam = 5, [3, 4], 0.1
print(loss + lam * sum(w*w for w in weights))`,
  'lasso-regression': `loss, weights, lam = 5, [-2, 3], 0.5
print(loss + lam * sum(abs(w) for w in weights))`,
  'elastic-net': `loss, weights, l2, l1 = 4, [1, -2], 0.1, 0.2
print(loss + l2 * sum(w*w for w in weights) + l1 * sum(abs(w) for w in weights))`,
  'polynomial-regression': `x = 3
print(1 + 2*x + 0.5*x*x)`,
  'squared-euclidean-distance': `x, y = [1, 2], [4, 6]
print(sum((a-b)**2 for a, b in zip(x, y)))`,
  'kmeans-sse': `points, center = [1, 5], 3
print(sum((point-center)**2 for point in points))`,
  'fcm-objective': `membership, distance, m = 0.8, 1, 2
print(round(membership**m * distance**2, 6))`,
  'fcm-membership': `distances, m = [1, 2], 2
# Для нулевого расстояния нужна отдельная ветка; здесь все расстояния > 0.
memberships = [1 / sum((di/dj)**(2/(m-1)) for dj in distances) for di in distances]
print(memberships)`,
  'silhouette': `a, b = 2, 5
print((b-a) / max(a, b))`,
  'mlp-hidden-activation': `from math import exp
inputs, weights = [1, 2], [0.5, 1]
z = sum(a*w for a, w in zip(inputs, weights))
print(z, round(1 / (1 + exp(-z)), 6))`,
  'mlp-forward-propagation': `from math import exp
sigmoid = lambda z: 1 / (1 + exp(-z))
hidden = sigmoid(2 * 0.5)
output = sigmoid(hidden * 2)
print(round(hidden, 6), round(output, 6))`,
  'mlp-cost': `from math import log
y, activation = 1, 0.8
loss = -(y*log(activation) + (1-y)*log(1-activation))
print(round(loss, 6))  # один выход, без регуляризации`,
  'backpropagation': `y, output, hidden, weight = 1, 0.8, 0.5, 2
delta_output = output - y
delta_hidden = weight * delta_output * hidden * (1-hidden)
gradient = hidden * delta_output
print([round(x, 6) for x in [delta_output, delta_hidden, gradient]])`,
  'gradient-checking': `loss = lambda w: w*w
w, eps = 3, 0.01
numeric = (loss(w+eps) - loss(w-eps)) / (2*eps)
analytic = 2*w
print(round(numeric, 6), abs(numeric-analytic) < 1e-9)`,
  'softmax': `from math import exp
scores = [2, 1, 0]
exponents = [exp(z-max(scores)) for z in scores]  # устойчивый softmax
probabilities = [e/sum(exponents) for e in exponents]
print([round(p, 6) for p in probabilities])
assert abs(sum(probabilities)-1) < 1e-12`,
  'tanh-activation': `from math import tanh
print(round(tanh(1), 6))`,
  'qkv-projections': `import numpy as np
X = np.array([[1., 2.]])  # [число токенов, размерность модели]
Wq = np.eye(2)
Wk = np.array([[0., 1.], [1., 0.]])
Wv = 2 * np.eye(2)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
print('Q:', Q, 'K:', K, 'V:', V)`,
  'attention-scores': `import numpy as np
Q = np.array([[1., 0.]])
K = np.array([[1., 0.], [0., 1.]])
scores = Q @ K.T / np.sqrt(K.shape[-1])
print(np.round(scores, 6))  # [1 query, 2 keys]`,
  'attention-softmax': `from math import exp
scores = [1., 2.]
exponents = [exp(s-max(scores)) for s in scores]
weights = [x/sum(exponents) for x in exponents]
print([round(w, 6) for w in weights])
assert abs(sum(weights)-1) < 1e-12`,
  'attention-output': `import numpy as np
# Входы выбраны так, чтобы softmax(scores) дал веса [0.67, 0.33].
Q = np.array([[1.]])
K = np.log(np.array([[0.67], [0.33]]))
V = np.array([[10., 0.], [0., 20.]])
scores = Q @ K.T / np.sqrt(K.shape[-1])
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
context = weights @ V
print(weights)
print(context)
assert np.allclose(context, [[6.7, 6.6]])`,
  'multi-head-attention': `import numpy as np
heads = [np.array([[1., 2.]]), np.array([[3., 4.]])]
W_output = np.eye(4)
joined = np.concatenate(heads, axis=-1)  # объединяем признаки, не токены
print(joined @ W_output)`,
  'sinusoidal-position': `from math import sin, cos
position, d_model = 1, 4
encoding = []
for i in range(d_model // 2):
    angle = position / 10000**(2*i/d_model)
    encoding.extend([sin(angle), cos(angle)])
print([round(value, 6) for value in encoding])`,
  'position-wise-ffn': `import numpy as np
x = np.array([[1., -2.]])
W1, W2 = np.eye(2), np.eye(2)
b1, b2 = np.zeros(2), np.zeros(2)
hidden = np.maximum(0, x @ W1 + b1)
print(hidden @ W2 + b2)`,
  'layer-normalization': `import numpy as np
x = np.array([[1., 2., 3.]])
gamma, beta, epsilon = 1., 0., 1e-5
mean = x.mean(axis=-1, keepdims=True)
variance = x.var(axis=-1, keepdims=True)
normalized = gamma * (x-mean) / np.sqrt(variance+epsilon) + beta
print(np.round(normalized, 6))`,
  'masked-lm-loss': `from math import log
correct_token_probabilities = [0.8]  # только позиции, выбранные для MLM
loss = -sum(log(p) for p in correct_token_probabilities)
print(round(loss, 6))`,
  'causal-lm': `from math import log, prod
conditional_probabilities = [0.5, 0.4, 0.2]
probability = prod(conditional_probabilities)
loss = -sum(log(p) for p in conditional_probabilities)
print(round(probability, 6), round(loss, 6))
assert abs(loss + log(probability)) < 1e-12`,
  'cosine-similarity': `from math import sqrt
u, v = [1, 2], [2, 1]
dot = sum(a*b for a, b in zip(u, v))
norm = lambda x: sqrt(sum(a*a for a in x))
similarity = dot / (norm(u)*norm(v))
print(round(similarity, 6))`
};
