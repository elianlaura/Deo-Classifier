# Jupyter Notebooks and python script for the Viva Bem Software Deliveries

##Abstract

The Alimentary Activity Classification project implements a deep learning model based on an Attention-enhanced Bidirectional Gated Recurrent Unit (BiGRU) architecture for sequence classification tasks. The code consists of two main scripts: cabigru_deo_train.py for training and cabigru_deo_test.py for evaluating the model. The training script processes a time-series dataset, applies a feature extraction pipeline, and trains a BiGRU model with self-attention to improve classification accuracy. It uses categorical focal loss to handle class imbalances and custom callbacks to track performance. The Viva-Bem dataset used involves activity recognition, particularly distinguishing between different actions such as eating, drinking, or other movements, as inferred from the dataset name (eatdrinkanother). The testing script loads the best-trained model, performs inference on the test set, and evaluates performance using metrics such as balanced accuracy, F1-score, and Cohen's Kappa. The results, including loss curves and evaluation reports, are saved for further analysis. This model is suitable for applications in human activity recognition, gesture classification, and other sequential classification tasks.

Files in This Repository
cabigru_deo_train.py: This script trains the Attention-BiGRU model on a specified dataset.
cabigru_deo_test.py: This script evaluates the trained model on a test dataset.


## Getting Started

**The Viva Bem software delivery** is a Jupyter Notebook. This material encapsulates all the process steps and important details.

Before you running our code, you must go to the [/data) and download the corresponding dataset in the path: data/. 

[Jupyter notebooks](https://jupyter.org/) allow a high-level of interactive learning, as code, text description, and visualization combined in one place.

## Prerequisites

You will require Jupyter notebook to run this code.

Install Jupyter by
download from the [official website](https://jupyter.org/), or
using automatic installer of, e.g., [Anaconda](https://www.anaconda.com/distribution/).

Start the Jupyter server.


### Installation of the Conda Environment

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
Open command prompt
Run the following code
`conda update --all`
`conda create --name <env_name> python=3.9 --file requirements.txt`
`conda activate <env_name>`

In the new conda environment, install TensorFlow (2.11 - 2.14) with GPU support according to the [official website] (https://www.tensorflow.org/install/pip)

To finish the installation of jupyter extensions, run:
`jupyter contrib nbextension install --user`

Select the folder with notebooks and run `jupyter notebook`

### Dependencies

matplotlib>=3.4.2
mlxtend>=0.23.1
numpy>=1.22.3
pandas>=1.2.4
scikit-learn>=1.3.2
scipy>=1.7.0
seaborn>=0.11.1
tensorboard>=2.5.0
tensorflow-estimator>=2.5.0
tensorflow-gpu>=2.5.0

### Running Jupyter Notebook


To run Jupyter Notebook, open an Anaconda prompt and navigate to the directory in which the cloned repository is. By default Jupyter is setup to open your home directory.

 Once you are in the correct folder, run Jupyter:

**jupyter notebook** or **jupyter-notebook** (depending on your operating system).

This should open Jupyter Notebooks in a browser window. On occasion, Jupyter may not be able to open a window and will give you a URL to paste in your browser. Please do so, if required.

### Running the python script

To run the Python script, open an Anaconda prompt and navigate to the directory where the cloned repository is located. Next, go into the <code>src</code> folder to run the following command with the parameter values that gave us the best performance:

python3 heart_rate_flag.py


## Support
For further assistance please contact us.