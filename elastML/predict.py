import os

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

models = {
    "LR": LinearRegression,
    "KNN": KNeighborsRegressor,
    "SVR": SVR,
    "GPR": GaussianProcessRegressor,
    "RF": RandomForestRegressor,
    "GBM": GradientBoostingRegressor,
    "KRR": KernelRidge,
    "MLP": MLPRegressor,
}

coss_val_parameters_bulk = {
    "LR": {"fit_intercept": False},
    "KNN": {"n_neighbors": 4},
    "SVR": {"C": 100.0, "degree": 3, "kernel": "poly"},
    "GPR": {"kernel": RBF(length_scale=1) + WhiteKernel(noise_level=1)},
    "RF": {"max_features": "sqrt", "n_estimators": 200},
    "GBM": {"learning_rate": 0.1, "n_estimators": 200},
    "KRR": {"alpha": 0.1, "degree": 1, "gamma": 0.1, "kernel": "rbf"},
    "MLP": {
        "activation": "tanh",
        "early_stopping": True,
        "hidden_layer_sizes": [150],
        "learning_rate": "adaptive",
    },
}


coss_val_parameters_shear = {
    "LR": {"fit_intercept": False},
    "KNN": {"n_neighbors": 4},
    "SVR": {"C": 100.0, "degree": 3, "kernel": "poly"},
    "GPR": {"kernel": RBF(length_scale=1) + WhiteKernel(noise_level=1)},
    "RF": {"max_features": None, "n_estimators": 200},
    "GBM": {"learning_rate": 0.1, "n_estimators": 200},
    "KRR": {"alpha": 0.01, "degree": 1, "gamma": 0.1, "kernel": "rbf"},
    "MLP": {
        "activation": "tanh",
        "early_stopping": True,
        "hidden_layer_sizes": [150],
        "learning_rate": "adaptive",
    },
}


def model_predict(predict_data, target, prop="Bulk", model_name="LR", print_mae=True):
    model_path = os.path.join(
        os.path.dirname(__file__), "{}_models/model_{}".format(prop, model_name)
    )
    model = joblib.load(model_path)
    if model_name == "GPR":
        target_transform_path = os.path.join(
            os.path.dirname(__file__), "transform/Transform_GPR_{}".format(prop)
        )
        target_transform = joblib.load(target_transform_path)

    predict = model.predict(predict_data).reshape(-1, 1)

    if model_name == "GPR" and print_mae:
        predict = target_transform.inverse_transform(predict)
        test_mae = mean_absolute_error(target.values.ravel(), predict.ravel())
        print("mae error: {} GPa".format(test_mae))
        return predict

    elif model_name == "GPR" and not print_mae:
        predict = target_transform.inverse_transform(predict)
        return predict

    elif model_name != "GPR" and print_mae:
        test_mae = mean_absolute_error(target.values.ravel(), predict.ravel())
        print("mae error: {} GPa".format(test_mae))
        return predict

    else:
        return predict
