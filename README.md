# PGCFC-Net: Periodic Dynamic Graph Meets Closed-Form Continuous-Time Learning for Traffic Flow Forecasting


# Data Preparation

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1yEifxrjy7soh0c0ztzksKGLF93Y8VcZ1/view?usp=drive_link).


Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

# Requirements

Python 3.6.5, Pytorch 2.10.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
python run.py --dataset {DATASET_NAME} 
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`

such as `python run.py --dataset PEMSD4`



