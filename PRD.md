# Program Requirement Document (PRD)

## Project Goal
Classification of the Iris dataset into 4 distinct categories, including a synthetic 4th category ("Iris-Unknown").

## Data Requirements
- **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width).
- **Samples:** 200 total samples (150 original, 50 synthetic).
- **Train-Test Split:** 80% training / 20% testing split as per course guidelines.

## Model Requirements
- **Classifier:** `SGDClassifier` from `scikit-learn`.
- **Training Strategy:** Track loss convergence per iteration using `partial_fit`.
- **Metrics:** Generate a 4x4 Confusion Matrix for classification performance analysis.

## Output Requirements
- **Files:** 
    - `confusion_matrix.png`: A 4x4 heatmap of prediction results.
    - `convergence_plot.png`: A line plot of Loss vs. Iterations.
    - `report.md`: Summary of dataset, model, and results.
    - `iris_extended.csv`: The complete extended dataset.
