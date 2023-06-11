# Libraries
import os
import json
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Project
from prep import ScrewData

# Set font defaults for all plots
plt.rcParams["font.family"] = "Times New Roman"

# Load screw data from json files at path
screw_data = ScrewData(path="data/")
torque, labels = screw_data.get_data()

# Get some information about the data
unique_values, counts = np.unique(labels, return_counts=True)
print("Unique Values:", unique_values)
print("Counts:", counts)


# # # 1 - F I G U R E   1 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Chart of repeated screwing (ratio of OK to NOK)

# List of 25 screw runs (timestamp indicates the screwing order)
# The work piece was chosen at random with no specific reason
# File names are specific screwing operations, indicated by the numeric portion
filename_list = [
    "Ch_000001601937.json",
    "Ch_000001601941.json",
    "Ch_000001601945.json",
    "Ch_000001601949.json",
    "Ch_000001601953.json",
    "Ch_000001601957.json",
    "Ch_000001601961.json",
    "Ch_000001601965.json",
    "Ch_000001601969.json",
    "Ch_000001601973.json",
    "Ch_000001601977.json",
    "Ch_000001601981.json",
    "Ch_000001601985.json",
    "Ch_000001601989.json",
    "Ch_000001601993.json",
    "Ch_000001601997.json",
    "Ch_000001602001.json",
    "Ch_000001602005.json",
    "Ch_000001602009.json",
    "Ch_000001602013.json",
    "Ch_000001602017.json",
    "Ch_000001602021.json",
    "Ch_000001602025.json",
    "Ch_000001602029.json",
    "Ch_000001602033.json",
]


# Load data of one work piece (one hole x 25 screw runs)
# Two lists to store torque and angle data for each run
data_runs_torque = []
data_runs_angle = []
for file in filename_list:
    with open(f"data/{file}", "r") as f:
        # Load screwing data from JSON
        screw_run = json.load(f)

        # Create lists to store run-specific torque and angle data
        data_run_torque = []
        data_run_angle = []
        for step in screw_run["tightening steps"]:
            # Append data for each tightening step
            data_run_torque.append(step["graph"]["torque values"])
            data_run_angle.append(step["graph"]["angle values"])

        # Flatten the nested list for each run
        data_run_torque = [item for sublist in data_run_torque for item in sublist]
        data_run_angle = [item for sublist in data_run_angle for item in sublist]

        # Add the run's data to the master lists
        data_runs_torque.append(data_run_torque)
        data_runs_angle.append(data_run_angle)

# Get a list of 25 colors from a color map
cmap = cm.get_cmap("viridis")  # choose the colormap
colors = cmap(np.linspace(0, 1, 25))
colors = [mcolors.to_hex(c) for c in colors]

# Plot screw runs
# Set figure size based on desired dimensions in centimeters
width_cm = 11.7  # width in centimeters
height_cm = 7.5  # height in centimeters
width_in = width_cm / 2.54  # conversion to inches
height_in = height_cm / 2.54  # conversion to inches
fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in), constrained_layout=True)

# Loop over each run's data and plot it
for i, (torque, angle) in enumerate(zip(data_runs_torque, data_runs_angle)):
    ax.plot(angle, torque, color=colors[i], linewidth=0.75, alpha=0.75)

# Style axis
ax.set_ylim(0, 1.6)
ax.set_xlim(0, 2000)
ax.grid(color="silver")
ax.set_ylabel("Torque [in Nm]")
ax.set_xlabel("Angle [in degree]")
ax.set_title("Comparison repeated tightening operations on the same work piece", pad=10)

# Add color bar
norm = mcolors.Normalize(0, 25)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax)

# Show plot
plt.savefig("images/figure_1.png", format="png")
plt.show()


# # # 2 - F I G U R E   2 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Plot the effect of the cycle number to the label

# Collect labels from each data file
labels = {}
file_names = os.listdir("data/")

# Loop through each file in the data directory
for file in file_names:
    with open(f"data/{file}", "r") as f:
        screw_run = json.load(f)

    # Extract identifying code and result
    dmc = screw_run["id code"]
    label = screw_run["result"]

    # Store labels by identifying code
    if dmc not in list(labels.keys()):
        labels[dmc] = [label]
    else:
        labels[dmc].append(label)

# Create DataFrame for label counts
label_counts = pd.DataFrame(columns=["dmc"] + [f"run_{i}" for i in range(1, 26)])

# Populate DataFrame with label counts by identifying code and run
for key, value in labels.items():
    curr_len = len(label_counts)
    label_counts.loc[curr_len] = [f"{key}_1st"] + value[0::2]
    label_counts.loc[curr_len + 1] = [f"{key}_2nd"] + value[1::2]

# Compute counts of 'OK' and 'NOK' labels
counts = label_counts.apply(lambda col: col.value_counts()).tail(2)
counts_OK = counts.T["OK"][1:].values / 200
counts_NOK = np.nan_to_num(counts.T["NOK"][1:].values) / 200

# Create bar plot
width_cm = 11.7  # width in centimeters
height_cm = 7.5  # height in centimeters
width_in = width_cm / 2.54
height_in = height_cm / 2.54
fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in), constrained_layout=True)

x = np.arange(len(counts_OK))  # the label locations
bar_width = 0.8

# Create stacked bars for 'OK' and 'NOK' label counts
ax.bar(x, counts_NOK, bar_width, color="firebrick", label="NOK")
ax.bar(x, counts_OK, bar_width, bottom=counts_NOK, color="forestgreen", label="OK")

# Add labels and title
ax.set_xlabel("Number of the respective screwing cycle")
ax.set_ylabel("Relative distribution of labels")
ax.set_title("Effect of the cycle number on the success of the screw run")

# Add legend and set axis limits
ax.legend(loc="upper left")
ax.set_xlim(-0.5, 24.5)
ax.set_ylim(0.0, 1.0)

# Show the plot
plt.savefig("images/figure_2.png", format="png")
plt.show()


# # # 3 - F I G U R E   3 # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FChart of repeated screwing (ratio of OK to NOK)

# List of 10 screw runs selected as examples of OK and NOK runs
filename_list = [
    # OK
    "Ch_000001601154.json",
    "Ch_000001598083.json",
    "Ch_000001597096.json",
    "Ch_000001596745.json",
    "Ch_000001597943.json",
    # NOK
    "Ch_000001602029.json",
    "Ch_000001596383.json",
    "Ch_000001602725.json",
    "Ch_000001596358.json",
    "Ch_000001603069.json",
]


# Load data of one work piece (one hole x 25 screw runs)
# Two lists to store torque and angle data for each run
data_runs_torque = []
data_runs_angle = []
data_runs_label = []
for file in filename_list:
    with open(f"data/{file}", "r") as f:
        # Load screwing data from JSON
        screw_run = json.load(f)

        # Create lists to store run-specific torque and angle data
        data_run_torque = []
        data_run_angle = []
        for step in screw_run["tightening steps"]:
            # Append data for each tightening step
            data_run_torque.append(step["graph"]["torque values"])
            data_run_angle.append(step["graph"]["angle values"])

        # Flatten the nested list for each run
        data_run_torque = [item for sublist in data_run_torque for item in sublist]
        data_run_angle = [item for sublist in data_run_angle for item in sublist]

        # Add the run's data to the master lists
        data_runs_torque.append(data_run_torque)
        data_runs_angle.append(data_run_angle)
        data_runs_label.append(step["result"])


# Get a list of 25 colors from a color map
cmap = cm.get_cmap("viridis")  # choose the colormap
colors = cmap(np.linspace(0, 1, 25))
colors = [mcolors.to_hex(c) for c in colors]

# Plot screw runs
# Set figure size based on desired dimensions in centimeters
width_cm = 11.7  # width in centimeters
height_cm = 7.5  # height in centimeters
width_in = width_cm / 2.54  # conversion to inches
height_in = height_cm / 2.54  # conversion to inches
fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in), constrained_layout=True)

# Loop over each run's data and plot it
for i, (torque, angle, label) in enumerate(
    zip(data_runs_torque, data_runs_angle, data_runs_label)
):
    color = "firebrick" if label == "NOK" else "forestgreen"
    ax.plot(angle, torque, color=color, linewidth=0.75, alpha=0.75)

# Style axis
ax.set_ylim(0, 1.6)
ax.set_xlim(0, 2000)
ax.grid(color="silver")
ax.set_ylabel("Torque [in Nm]")
ax.set_xlabel("Angle [in degree]")
ax.set_title("Visualization of normal an anormal observations", pad=10)

# Show plot
plt.savefig("images/figure_3.png", format="png")
plt.show()
