import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = list(iris.target_names)

# 2. Add a synthetic 4th category (Iris-Unknown)
# We take a portion of the data and add significant noise to create a new class
np.random.seed(42)
noise = np.random.normal(0, 1.5, (50, 4))
X_unknown = X[:50] + noise
y_unknown = np.full((50,), 3) # Label 3 for the 4th category
target_names.append('unknown')

X_extended = np.vstack([X, X_unknown])
y_extended = np.concatenate([y, y_unknown])

# 3. Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_extended, y_extended, test_size=0.20, random_state=42)

# 4. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SGDClassifier and track loss
model = SGDClassifier(loss='log_loss', max_iter=1, random_state=42)
classes = np.unique(y_extended)
losses = []
n_iterations = 100

for i in range(n_iterations):
    model.partial_fit(X_train, y_train, classes=classes)
    y_prob = model.predict_proba(X_train)
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(np.eye(len(classes))[y_train] * np.log(y_prob), axis=1))
    losses.append(loss)

# 6. Final Accuracy Score
y_pred = model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Accuracy: {final_accuracy:.4f}")

# 7. Generate 4x4 Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('4x4 Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# 8. Generate Convergence Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iterations + 1), losses, label='Training Loss', color='orange')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Convergence Plot (4 Categories)')
plt.legend()
plt.grid(True)
plt.savefig('convergence_plot.png')
plt.close()

with open('accuracy.txt', 'w') as f:
    f.write(f"{final_accuracy:.4f}")
