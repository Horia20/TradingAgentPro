from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model):

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=3
    )

    return calibrated