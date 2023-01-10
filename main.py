import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

eps = 0.001
C = 2
gamma = 2
tol = 1e-4


def compute_L(p, i1, i2):
    if p.Y[i1] != p.Y[i2]:
        return max(0, p.A[i2]-p.A[i1])
    else:
        return max(0, p.A[i2] + p.A[i1] - p.C)


def compute_H(p, i1, i2):
    if p.Y[i1] != p.Y[i2]:
        return min(p.C, p.C + p.A[i2] - p.A[i1])
    else:
        return min(p.C, p.A[i2] + p.A[i1])


def simp_kernel(p, i1, i2):
    return i1.dot(i2)


def rbf_kernel(p, i1, i2):
    distance = np.linalg.norm(i1 - i2, axis=-1)

    K = np.exp(-gamma * (distance ** 2))

    return K


def kernel(p, i1, i2):
    return rbf_kernel(p, i1, i2)


def SVM(p, i):
    return np.dot(p.A * p.Y, kernel(p, p.X, p.X[i])) - p.B


def calculate_new_alpha2(alph2, y2, e1, e2, eta):
    return alph2 + y2 * (e1 - e2) / eta


def clip_alpha(new_alpha, L, H):
    if new_alpha <= L:
        return L
    elif new_alpha >= H:
        return H
    return new_alpha


def update_b(p, i1, i2, alpha_i1, alpha_i2):
    b1 = p.E[i1] + p.Y[i1]*(alpha_i1 - p.A[i1])*kernel(p, p.X[i1], p.X[i1]) + p.Y[i2]*(alpha_i2 - p.A[i2])*kernel(p, p.X[i1], p.X[i2]) + p.B
    b2 = p.E[i2] + p.Y[i1]*(alpha_i1 - p.A[i1])*kernel(p, p.X[i1], p.X[i2]) + p.Y[i2]*(alpha_i2 - p.A[i2])*kernel(p, p.X[i2], p.X[i2]) + p.B

    if 0 < alpha_i1 < p.C:
        p.B = b1
    elif 0 < alpha_i2 < p.C:
        p.B = b2
    else:
        p.B = (b1 + b2) / 2


def update_weight(p, i1, i2, a1, a2):
    p.W = p.W + p.Y[i1]*(a1 - p.A[i1])*p.X[i1] + p.Y[i2]*(a2 - p.A[i2])*p.X[i2]


def objective_function_at_a2(p, a2_val, i1, i2):
    k11 = kernel(p, p.X[i1], p.X[i1])
    k12 = kernel(p, p.X[i1], p.X[i2])
    k22 = kernel(p, p.X[i2], p.X[i2])

    s = p.Y[i1] * p.Y[i2]

    f1 = p.Y[i1] * (p.E[i1] + p.B) - p.A[i1] * k11 - s * p.A[i2] * k12
    f2 = p.Y[i2] * (p.E[i2] + p.B) - s * p.A[i1] * k12 - p.A[i2] * k22
    new_a1 = p.A[i1] + s * (p.A[i2] - a2_val)

    return new_a1 * f1 + a2_val * f2 +\
           0.5 * new_a1 * new_a1 * k11 +\
           0.5 * a2_val * a2_val * k22 +\
           s * a2_val * new_a1 * k12


def update_error_cache(p):
    for i in range(0, p.n):
        p.E[i] = SVM(p, i) - p.Y[i]


def take_step(p, i1, i2):
    if i1 == i2:
        return 0

    alph1 = p.A[i1]
    alph2 = p.A[i2]
    y1 = p.Y[i1]
    e1 = SVM(p, i1) - y1
    s = p.Y[i1] * p.Y[i2]

    L = compute_L(p, i1, i2)
    H = compute_H(p, i1, i2)
    if L == H:
        return 0

    y2 = p.Y[i2]
    e2 = SVM(p, i2) - y2

    k11 = kernel(p, p.X[i1], p.X[i1])
    k12 = kernel(p, p.X[i1], p.X[i2])
    k22 = kernel(p, p.X[i2], p.X[i2])
    eta = k11 + k22 - (2 * k12)
    if eta > 0:
        a2 = clip_alpha(calculate_new_alpha2(alph2, y2, e1, e2, eta), L, H)
    else:
        Lobj = objective_function_at_a2(p, L, i1, i2)
        Hobj = objective_function_at_a2(p, H, i1, i2)
        if Lobj < Hobj - eps:
            a2 = L
        elif Lobj > Hobj + eps:
            a2 = H
        else:
            a2 = alph2

    if abs(a2-alph2) < eps*(a2+alph2+eps):
        return 0

    a1 = alph1 + s * (alph2 - a2)
    update_b(p, i1, i2, a1, a2)
    update_weight(p, i1, i2, a1, a2)
    p.A[i1] = a1
    p.A[i2] = a2
    update_error_cache(p)
    return 1


def non_bound_examples(p):
    return [i for i in range(0, p.n) if p.A[i] != 0 and p.A[i] != p.C]


def second_choice_heuristic(p, i2):
    non_bounds = non_bound_examples(p)
    if i2 in non_bounds:
        non_bounds.remove(i2)
    m = non_bounds[0]
    if p.E[i2] >= 0:
        for i in non_bounds:
            if p.E[i] < p.E[m]:
                m = i
        return m
    else:
        for i in non_bounds:
            if p.E[i] > p.E[m]:
                m = i
        return m


def examine_examples(p, i2):
    y2 = p.Y[i2]
    alph2 = p.A[i2]
    E2 = SVM(p, i2) - y2
    r2 = E2 * y2
    if (r2 < -p.TOL and alph2 < p.C) or (r2 > p.TOL and alph2 > 0):
        non_bounds = non_bound_examples(p)
        if len(non_bounds) > 1:
            i1 = second_choice_heuristic(p, i2)
            if take_step(p, i1, i2):
                return 1

        random.shuffle(non_bounds)
        for i1 in non_bounds:
            if take_step(p, i1, i2):
                return 1

        l = [i for i in range(0, p.n)]
        random.shuffle(l)
        for i1 in l:
            if take_step(p, i1, i2):
                return 1

    return 0


class PrimalProblem:
    def __init__(self, x, y, c, t):
        self.TOL = t
        self.C = c
        self.X = x
        self.Y = y
        self.B = 0
        n, m = x.shape
        self.n = n
        self.A = np.zeros(n)
        self.E = np.zeros(n)
        self.W = np.zeros(m)


def SMO(x, y, c, tolerance):
    p = PrimalProblem(x, y, c, tolerance)

    num_changed = 0
    examine_all = 1
    update_error_cache(p)

    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
             for i in range(p.n):
                num_changed += examine_examples(p, i)
        else:
            non_bound = non_bound_examples(p)
            for i in non_bound:
                num_changed += examine_examples(p, i)

        if examine_all:
            examine_all = 0
        elif not num_changed:
            examine_all = 1

    return p


def predict_svm(model, X):
    predictions = []
    for x in X:
        prediction = np.dot(model.A * model.Y, kernel(model, model.X, x)) - model.B
        predictions.append(prediction)
    return predictions


def predict(X, models):
    predictions = []
    for model in models:
        prediction = predict_svm(model, X)
        predictions.append(prediction)

    return [prediction.index(max(prediction)) for prediction in list(zip(predictions[0], predictions[1], predictions[2]))]


def SVM_train(X, y, C, tolerance):
    num_classes = len(set(y))
    models = []

    for i in range(num_classes):
        # Create a dataset for the current class
        X_class = []
        y_class = []
        for j in range(len(X)):
            X_class.append(X[j])
            if y[j] == i:
                y_class.append(1)
            else:
                y_class.append(-1)

        # Train a binary SVM on the class dataset
        model = SMO(np.array(X_class), np.array(y_class), C, tolerance)
        models.append(model)
    return models


def scatter_plot_matrix(iris):
    X = iris.data
    y = iris.target
    # Set the colors for each flower species
    colors = {iris.target_names[0]: 'red', iris.target_names[1]: 'green', iris.target_names[2]: 'blue'}

    # Create a figure with a subplot for each pair of features
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    # Plot the diagonal subplots
    for i, j in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

    # Plot the off-diagonal subplots
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]:
        axs[i, j].scatter(X[:, i], X[:, j], c=[colors[iris.target_names[k]] for k in y])
        axs[i, j].set_xticks([])
        axs[i, j].set_xlabel(iris.feature_names[i])
        axs[i, j].set_yticks([])
        axs[i, j].set_ylabel(iris.feature_names[j])

    # Show the plot
    patch0 = mpatches.Patch(color=colors[iris.target_names[0]], label=iris.target_names[0])
    patch1 = mpatches.Patch(color=colors[iris.target_names[1]], label=iris.target_names[1])
    patch2 = mpatches.Patch(color=colors[iris.target_names[2]], label=iris.target_names[2])
    fig.legend(handles=[patch0, patch1, patch2])
    plt.show()


def main():
    # Load the iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    #  scatter plot matrix
    scatter_plot_matrix(iris)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)

    models = SVM_train(X_train, y_train, C, tol)

    # Get predictions for the test set
    predictions = predict(X_test, models)

    # Create a multi-class confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(conf_matrix)
    print(accuracy)

    # Calculate the sensitivity for each class
    sensitivity = []
    for i in range(conf_matrix.shape[0]):
        true_positive = conf_matrix[i, i]
        false_negative = conf_matrix[i, :].sum() - true_positive
        sensitivity.append(true_positive / (true_positive + false_negative))

    print(sensitivity)

if __name__ == '__main__':
    main()
