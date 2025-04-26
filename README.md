# Airbnb Price Prediction

## How to Build and Run the Code

This project uses a Makefile to install dependencies, run the model, and generate visualizations.

First, make sure you have Python installed. Then, in your terminal:

1. Clone the repository (if you haven't already).
2. Navigate into the project folder.
3. Run the following command:
    ```bash
    make all
    ```

This will:
- Install all dependencies listed in `requirements.txt`
- Run the Random Forest model (`Final_project_RandomForest.py`)
- Generate plots (`visualizing.py`)

## Individual Commands

You can also run tasks separately:

- Install dependencies:
    ```bash
    make install
    ```

- Run the model:
    ```bash
    make run
    ```

- Regenerate visualizations:
    ```bash
    make visualize
    ```

- Clean up Python cache files:
    ```bash
    make clean
    ```

## Project Structure

- `Final_project_RandomForest.py` — Code to train and evaluate the Random Forest model.
- `visualizing.py` — Code to generate visualizations of the results.
- `requirements.txt` — Python dependencies.
- `Makefile` — Automation for building, running, and cleaning the project.
