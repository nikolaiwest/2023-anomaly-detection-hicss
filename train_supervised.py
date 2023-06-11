# Libraries
import pandas as pd
from datetime import datetime as dt

# Project
from models.supervised import (
    CustomCNN as CNN,
    CustomEncoder as Encoder,
    CustomLSTM as LSTM,
    CustomRandomForest as RandomForest,
)
from utilities import (
    print_status,
    run_cross_validation,
    get_metrics,
    columns_order,
)
from prep import ScrewData

# Load screw data from json files at path
screw_data = ScrewData(path="data/")
torque, labels = screw_data.get_data()
print_status("Screw driving data sucessfully loaded")

# Create a result dataframe to later store as csv for accessibility
metrics_rf = pd.DataFrame(columns=columns_order)
metrics_lstm = pd.DataFrame(columns=columns_order)
metrics_cnn = pd.DataFrame(columns=columns_order)
metrics_encoder = pd.DataFrame(columns=columns_order)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        S U P E R V I S E D   M O D E L S                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 1. CNN classifier : build with CNNClassifier from sktime_dl.deeplearning
# 2. Encoder classifier : build with EncoderClassifier from sktime_dl.deeplearning
# 3. LSTM classifier : build with keras.LSTM (750 -> 64 -> 31 -> 1)
# 4. Random Forest classifier : build with RandomForestClassifier from sklearn.ensemble


# # # 1 - C N N    C L A S S I F I E R # # # # # # # # # # # # # # # # # # # # # # #

# CNN: Start
print_status("CNN - Starting training and evaluation")
start_cnn = dt.now()

for f, x_train, y_train, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Reshaping, since CNN expects 3D input: [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Get model
    cnn = CNN(nb_epochs=10, verbose=True)

    # Fit model
    cnn.fit(x_train, y_train)

    # Apply model to test data
    predictions_cnn = cnn.predict(x_test)

    # Evaluate the prediction using classification metrics
    metrics_dict = get_metrics(y_test, predictions_cnn)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_cnn = metrics_cnn.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"CNN- Finished training and evaluation in {dt.now() - start_cnn}")
metrics_cnn.to_csv("results/result_df_cnn.csv")


# # # 2 - E N C O D E R     C L A S S I F I E R # # # # # # # # # # # # # # # # # # # #

# Encoder: Start
print_status("Encoder - Starting training and evaluation")
start_encoder = dt.now()

for f, x_train, y_train, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Reshaping, since Encoder expects 3D input: [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Get model
    encoder = Encoder(nb_epochs=10, verbose=True)

    # Fit model
    hist_encoder = encoder.fit(x_train, y_train)

    # Apply model to test data
    predictions_encoder = encoder.predict(x_test)

    # Evaluate the prediction using classification metrics
    metrics_dict = get_metrics(y_test, predictions_encoder)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_encoder = metrics_encoder.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"Encoder- Finished training and evaluation in {dt.now() - start_encoder}")
metrics_encoder.to_csv("results/result_df_encoder.csv")


# # # 3 - L S T M   C L A S S I F I E R # # # # # # # # # # # # # # # # # # # # # #

# LSTM: Start
print_status("LSTM - Starting training and evaluation")
start_lstm = dt.now()

# Ten-fold cross validation
for f, x_train, y_train, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Reshaping, since LSTM expects 3D input: [samples, timesteps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Get model
    lstm = LSTM(input_shape=(x_train.shape[1], 1), dropout=0.0)

    # Fit model with custom weights
    class_weights = {0: 0.16, 1: 0.84}  # assuming class 1 is the minority class
    lstm.fit(x_train, y_train, epochs=100, batch_size=32, class_weight=class_weights)

    # Apply model to test data
    predictions_lstm = lstm.predict(x_test)

    # Evaluate the prediction using classification metrics
    metrics_dict = get_metrics(y_test, predictions_lstm)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_lstm = metrics_lstm.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(f"LSTM - Finished training and evaluation in {dt.now() - start_lstm}")
metrics_lstm.to_csv("results/result_df_lstm__.csv")


# # # 4 - R A N D O M   F O R E S T # # # # # # # # # # # # # # # # # # # # # # # #

# Random Forest: Start
print_status("Random Forest - Starting training and evaluation")
start_rf = dt.now()

# Ten-fold cross validation
for f, x_train, y_train, x_test, y_test in run_cross_validation(torque, labels, k=10):
    # Get model
    random_forest = RandomForest(n_estimators=100, random_state=42)

    # Fit model
    random_forest.fit(x_train, y_train)

    # Apply model to test data
    predictions_rf = random_forest.predict(x_test)

    # Evaluate the prediction using classification metrics
    metrics_dict = get_metrics(y_test, predictions_rf)

    # Add results to the metric dataframe
    metrics_dict["Fold"] = f
    metrics_rf = metrics_rf.append(metrics_dict, ignore_index=True)

## Finish and save results
print_status(
    f"Random Forest - Finished training and evaluation in {dt.now() - start_rf}"
)
metrics_rf.to_csv("results/result_df_rf.csv")
