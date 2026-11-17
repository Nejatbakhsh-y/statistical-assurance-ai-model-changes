from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_baseline(x,y,seed):
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed))
    return model.fit(x,y)
