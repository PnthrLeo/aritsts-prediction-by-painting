# Aritsts Prediction By Paintings

Simple app with GUI for artists prediction by paintings via Neural Networks.

Neural Networks Models used in this app:
1. ResNet-50
2. EfficientNet-B4
3. SReT-S


## Installation and start

1. Install requirements (environment was tested for Python 3.7.17 and CUDA toolkit 10.2)

    ```bash
    pip install -r requirements.txt
    ```

2. Pretrain models' weights via **./notebooks/model_training.ipynb**
3. Save models' weights to **./data/model_weights**
4. Change models' weights paths in **./contollers/_main.py if needed**
5. Save **artists.csv** to **./data** (can be generated in **./notebooks/model_training.ipynb**)

6. Start

    ```bash
    python app.py
    ```

## Usage
TODO!!!
