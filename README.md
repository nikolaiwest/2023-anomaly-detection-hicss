# Anomaly Detection for Industrial Screw Connections: HICSS 2023
This repository contains results of following conference publication: 
## Publication
#### Title
A comparative study of machine learning approaches for anomaly detection in industrial screw driving data

#### Authors
- Nikolai West <sup> [ORCID](https://orcid.org/0000-0002-3657-0211) </sup>
- Jochen Deuse <sup> [ORCID](https://orcid.org/0000-0003-4066-4357) </sup>

#### Abstract 
This paper investigates the application of Machine Learning (ML) approaches for anomaly detection in screw driving operations, a pivotal process in manufacturing. Leveraging a novel, open-access real-world dataset, we explore the efficacy of several unsupervised and supervised ML models. Among unsupervised models, DBSCAN demonstrates superior performance with an accuracy of 96.68% and a Macro F1 score of 90.70%. Within the supervised models, the Random Forest model excels, achieving an accuracy of 99.02% and a Macro F1 score of 98.36%. These results not only underscore the significant potential of ML in boosting manufacturing quality and efficiency, but also highlight the challenges in their practical deployment. This research encourages further investigation and refinement of ML techniques for industrial anomaly detection, thereby contributing to the advancement of resilient, efficient, and sustainable manufacturing processes. The entire analysis, comprising the complete data set as well as the Python-based scripts are made publicly available via a dedicated repository. This commitment to open science aims to support the practical application and future adaptation of our work in to support business decisions in quality management and the manufacturing industry. 

#### Status
- _Submitted for review_

## Repository 
#### Directory Structure

2023-anomaly-detection-hicss
├── data # Dataset for the project

├── images # Figures from the publication

├── models 

│ ├── supervised # Supervised learning models

│ └── unsupervised # Unsupervised learning models

├── results # Final results and evaluation metrics (as .csv)

├── eval.py # Script for evaluating the models performances

├── LICENSE.md

├── prep.py # Script for preprocessing the screw data

├── README.md 

├── requirements.txt

├── train_supervised.py # Training script for supervised models

├── train_unsupervised.py # Training script for unsupervised models

├── utilities.py # Utility functions used across the project

└── vis.py # Script to create visualizations in images/

#### Getting Started
To get started with the project:
1. Clone this repository to your local machine.

```git clone https://github.com/nikolaiwest/2023-anomaly-detection-hicss.git```

```cd 2023-anomaly-detection-hicss```

2. Install the required dependencies using python `3.7.9`:
```pip install -r requirements.txt```

## Usage
#### Preprocessing the data
To load and preprocess the raw data, use the `prep.py` script, e.g. like this: 

```screw_data = ScrewData(path="data/")```

```torque, labels = screw_data.get_data()```

#### Training the models
To train the models, use the `train_supervised.py` and `train_unsupervised.py` scripts:

```python train_supervised.py```

```python train_unsupervised.py```

#### Evaluating the models
Training the models already comes with an evaluation. The results are stored as `.csv` in `results/`. 
However, you can get a simple comparison of all models by running: 

```python eval.py```

#### Creating visualizations
If you wish to recreate the visualizations from the respective paper, use `vis.py` that also saves the figures at `images/`:

```python vis.py```

## Contributing
We welcome contributions to this repository. If you have a feature request, bug report, or proposal, please open an issue. If you wish to contribute code, please open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.