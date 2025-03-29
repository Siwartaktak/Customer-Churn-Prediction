.PHONY: setup train test api clean

# Variables
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
TRAIN_DATA = C:/Users/MSI/Downloads/churn-bigml-80.csv
TEST_DATA = C:/Users/MSI/Downloads/churn-bigml-20.csv
OUTPUT_DIR = models
EXPERIMENT_NAME = churn_prediction

setup:
    python -m venv $(VENV)
    $(PIP) install -r requirements.txt

train:
    $(PYTHON) src/main.py --train-data $(TRAIN_DATA) --test-data $(TEST_DATA) --output-dir $(OUTPUT_DIR) --experiment-name $(EXPERIMENT_NAME)

test:
    $(PYTHON) -m pytest tests/

api:
    $(PYTHON) -m uvicorn src.api.main:app --reload

clean:
    rm -rf $(OUTPUT_DIR)/*.joblib
    rm -rf $(OUTPUT_DIR)/*.png
    rm -rf __pycache__
    rm -rf src/__pycache__
    rm -rf src/*/__pycache__