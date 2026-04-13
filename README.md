# Iris Classification Project (4 Categories)
#GroupID:biu-maho-ex01
#Submited: by Maor Gerafi & Hodaya Golomb 
## Overview
This project performs classification on an extended version of the classic Iris dataset. To meet specific course requirements, a synthetic 4th category ("Iris-Unknown") was added to the original 3-class dataset, resulting in 200 total samples.

## Project Structure
- `solution.py`: The main Python script that handles data preparation, model training (SGD), and visualization.
- `iris_extended.csv`: The processed dataset including the 4th category.
- `PRD.md`: Program Requirement Document.
- `report.md`: Final project report with results.
- `confusion_matrix.png`: A 4x4 heatmap of the classification results.
- `convergence_plot.png`: A graph showing loss reduction over 100 iterations.

## How to Run
1. Ensure you have Python installed.
2. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn`
3. Execute the script: `python solution.py`
