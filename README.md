# Next Word Prediction Pipeline

## Overview
This project aims to build Next Word Prediction pipeline using LSTM.

## Prerequisites
- Python 3.8 or higher

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train the model, run the following command:
```bash
python train.py
```


## API Endpoints

To start the Fastapi server:

```bash
uvicorn app.main:app --reload
```

1. Get : `localhost:8000/`
   
    Response:
    ```json
    {"message": "Welcome to the SVM Image Classification API"}
    ```

2. Get : `localhost:8000/predict?text={text to classify}`
   
    Response:
    ```json
    {"text": "Hello, world!", "prediction": "ham", "probability": [0.9964659742064316, 0.0035340257935684055]}
    ```

## Usage
To run the Fastapi server:
```bash
curl 'localhost:8000/predict?img_path=dataset_images%5Ccat%5Ccat_1.jpg'
```
This will output the predicted class for the given image.
