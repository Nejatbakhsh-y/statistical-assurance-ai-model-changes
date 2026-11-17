from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(seed, test_size):
    d = load_breast_cancer()
    xtr, xte, ytr, yte = train_test_split(
        d.data, d.target, test_size=test_size, stratify=d.target, random_state=seed
    )
    return xtr, xte, ytr, yte
