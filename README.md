# Iris Classifier

This is a small Python project that trains a classifier on the classic Iris dataset using scikit-learn.

The aim of the project was to practise:
- loading a dataset
- training a model
- evaluating results
- saving outputs in a clean project structure

## Project structure

iris-classifier/
├── notebooks/ # Jupyter notebook used for exploration
├── src/ # Training script
│ └── train.py
├── outputs/ # Generated outputs
│ └── confusion_matrix.png


## What the code does

- Loads the Iris dataset from scikit-learn  
- Splits the data into training and test sets  
- Trains a Decision Tree classifier  
- Prints the model accuracy  
- Saves a confusion matrix image to the `outputs` folder  

## How to run it

From the project root:

```bash
python src/train.py --test-size 0.2 --random-state 42
