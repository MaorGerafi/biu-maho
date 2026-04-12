# Project Plan: Iris Classification

1. **Data Preparation**: 
   - Load the standard Iris dataset.
   - Inject Gaussian noise into a subset of samples to create a 4th synthetic class ("Iris-Unknown").
   - Export the extended dataset to `iris_extended.csv`.
2. **Model Training**:
   - Split data into 80% training and 20% testing sets.
   - Use `StandardScaler` for feature normalization.
   - Train an `SGDClassifier` iteratively using `partial_fit` to monitor loss.
3. **Evaluation**:
   - Calculate final accuracy score on the test set.
   - Generate a 4x4 Confusion Matrix to visualize performance across all 4 classes.
4. **Reporting**:
   - Plot the convergence curve (Loss vs. Iterations).
   - Document all findings in `report.md` and `PRD.md`.
