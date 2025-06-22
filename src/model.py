from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def build_model():
    return Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(alpha=1.0))])


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R²:", r2_score(y_test, preds))
    return model, preds
