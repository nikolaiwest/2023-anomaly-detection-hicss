# Libraries
import numpy as np
import pandas as pd
from datetime import datetime as dt

# Project
from models.unsupervised import (
    CustomAutoencoder as Autoencoder,
    CustomDBSCAN as DBSCAN,
    CustomIsolationForest as IsolationForest,
    CustomLocalOutlierFactor as LocalOutlierFactor,
)
from utilities import (
    print_status,
    run_cross_validation,
    get_metrics,
    TqdmProgressCallback,
    columns_order,
)
from prep import ScrewData

# Load screw data from json files at path
screw_data = ScrewData(path="data/")
torque, labels = screw_data.get_data()
print_status("Screw driving data sucessfully loaded")

# Create a result dataframe to later store as csv for accessibility
metrics_ae = pd.DataFrame(columns=columns_order)
metrics_if = pd.DataFrame(columns=columns_order)
metrics_dbscan = pd.DataFrame(columns=columns_order)
metrics_lof = pd.DataFrame(columns=columns_order)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                      U N S U P E R V I S E D   M O D E L S                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 1. Autoencoder : build with keras.Sequential (128 -> 64 -> 32 -- 32 -> 64 -> 128)
# 2. DBSCAN : build with default DBSCAN from sklearn.cluster
# 3. Isolation Forest : build with default IsolationForest from sklearn.ensemble
# 4. Local Outlier Factor : build with default LocalOutlierFactor from sklearn.neighbors


# # # 1 - A U T O E N C O D E R # # # # # # # # # # # # # # # # # # # # # # # # # #

# AE: Start
print_status("Autoencoder - Starting training and evaluation")
start_ae = dt.now()

# Ten-fold cross validation
for f, x_train, y_train, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Remove NOK data from training set
    x_train = x_train[np.where(y_train == 0)]
    y_train = y_train[np.where(y_train == 0)]

    # Get model
    autoencoder = Autoencoder(len_target=screw_data.len_target)

    # Compile model
    autoencoder.compile(optimizer="adam", loss="mae")

    # Fit model
    history_ae = autoencoder.fit(
        x_train,
        x_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        shuffle=False,
        verbose=0,
        callbacks=[TqdmProgressCallback()],
    )

    # Apply model to test data
    predictions_ae = autoencoder.predict(x_test)

    # Compute Mean Squared Error for each prediction
    mse_ae = np.mean(np.power(x_test - predictions_ae, 2), axis=1)

    # Classify as anomalies those examples whose MSE is greater than the threshold
    predictions_ae_label = mse_ae > 0.0075

    # Convert anomalies array to int: 1 for True (anomaly), 0 for False (normal)
    predictions_ae_label = predictions_ae_label.astype(int)

    # Evaluate the prediction using classification metrics
    metrics_dict = get_metrics(y_test, predictions_ae_label)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_ae = metrics_ae.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"Autoencoder - Finished training and evaluation in {dt.now() - start_ae}")
metrics_ae.to_csv("results/result_df_ae.csv")


# # # 2 - D B S C A N # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# DBSCAN: Start
print_status("DBSCAN - Starting training and evaluation")
start_dbscan = dt.now()

# Ten-fold cross validation
for f, _, _, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Get model
    dbscan = DBSCAN(eps=3.5, min_samples=5)

    # Fit model and predict
    metrics_dict = dbscan.fit_and_predict(x_test, y_test)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_dbscan = metrics_dbscan.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"DBSCAN - Finished training and evaluation in {dt.now() - start_dbscan}")
metrics_dbscan.to_csv("results/result_df_dbscan.csv")


# # # 3 - I S O L A T I O N   F O R E S T # # # # # # # # # # # # # # # # # # # # #

# IF: Start
print_status("Isolation Forest - Starting training and evaluation")
start_if = dt.now()

# Ten-fold cross validation
for f, _, _, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Get model
    isolation_forest = IsolationForest(
        n_estimators=100,
        random_state=42,
        contamination=0.16,
    )

    # Fit model, predict and evaluate
    metrics_dict = isolation_forest.fit_and_predict(x_test, y_test)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_if = metrics_if.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(
    f"Isolation Forest - Finished training and evaluation in {dt.now() - start_if}"
)
metrics_if.to_csv("results/result_df_if.csv")


# # # 4 - L O C A L   O U T L I E R   F A C T O R # # # # # # # # # # # # # # # # #

# Ten-fold cross validation
print_status("LOF - Starting training and evaluation")
start_lof = dt.now()

# Cross validation
for f, _, _, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Get model
    lof = LocalOutlierFactor(n_neighbors=35, contamination=0.2)

    # Fit model and predict
    metrics_dict = lof.fit_and_predict(x_test, y_test)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_lof = metrics_lof.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"LOF - Finished training and evaluation in {dt.now() - start_lof}")
metrics_lof.to_csv("results/result_df_lof.csv")
