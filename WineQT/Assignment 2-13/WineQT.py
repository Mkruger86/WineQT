import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Assignment 2
datafile_url = "https://raw.githubusercontent.com/jpandersen61/Machine-Learning/refs/heads/main/WineQT.csv"
dataset = pd.read_csv(datafile_url)
# local_file_path = Path("WineQT/Assignment 2-13/WineQT.csv")
# if not local_file_path.exists():
#     dataset = pd.read_csv(datafile_url)
#     dataset.to_csv(local_file_path, index=False)
#     print(f"File downloaded and saved as {local_file_path}")
# else:
#     dataset = pd.read_csv(local_file_path)
#     print(f"File already exists. Loaded {local_file_path}")

# Assignment 3
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
print(dataset.head())

# Assignment 4
correlation_matrix = dataset.drop(columns=['Id']).corr()
heatmap_file_path = Path("WineQT/Assignment 2-13/correlation_matrix_heatmap.png")
if not heatmap_file_path.exists():
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_file_path)
    print(f"Heatmap saved as {heatmap_file_path}")
else:
    print(f"Heatmap already exists at {heatmap_file_path}")
## Dropping unnecessary columns based on correlation matrix analysis and the seaborn heatmap, which i greatly prefer for visualization.
dataset.drop(columns=['fixed acidity', 'free sulfur dioxide'], axis=1, inplace=True)
## Extract X and y. Since max appears to be 8 according to describe, we set > 6. Specific 6, 7 values is not prudent for this dataset when max value is 8
X = dataset.iloc[:, :-2]
y = dataset["quality"] > 6  

# Assignment 5+6
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print(f"X_train shape: {X_train.info()}, y_train shape: {y_train.info()}")
print(f"X_test shape: {X_test.info()}, y_test shape: {y_test.info()}")

# Assignment 7+8
mlp_clf = MLPClassifier(hidden_layer_sizes=[5], activation="relu", max_iter=10000, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_clf)

# Assignment 9+10
pipeline.fit(X_train, y_train)

# Assignment 11
accuracy_train=pipeline.score(X_train, y_train)
print(f"Training accuracy: {accuracy_train}")

accuracy_test=pipeline.score(X_test, y_test)
print(f"Test accuracy: {accuracy_test}")

# assignment 12, 1
dataset_copy = dataset.copy()

hidden_layer_sizes_list = [[5], [10], [5, 10], [10, 15, 20]]
activation_functions = ["relu", "tanh", "logistic"]

results_experiment_1 = []  

for hidden_layers in hidden_layer_sizes_list:
    for activation in activation_functions:
        print(f"Testing hidden_layer_sizes={hidden_layers}, activation={activation}")
        
        mlp_clf_exp1 = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=10000, random_state=42)
        pipeline_exp1 = make_pipeline(StandardScaler(), mlp_clf_exp1)
        
        pipeline_exp1.fit(X_train, y_train)
        
        accuracy_train_exp1 = pipeline_exp1.score(X_train, y_train)
        accuracy_test_exp1 = pipeline_exp1.score(X_test, y_test)
        print(f"Training accuracy: {accuracy_train_exp1}, Test accuracy: {accuracy_test_exp1}")
        
        results_experiment_1.append((hidden_layers, activation, accuracy_train_exp1, accuracy_test_exp1))

print("\nResults for different configurations (Assignment 12, 1):")
for hidden_layers, activation, train_acc, test_acc in results_experiment_1:
    print(f"hidden_layer_sizes={hidden_layers}, activation={activation}, Train Acc={train_acc}, Test Acc={test_acc}")

# assignment 12, 2
dataset_modified = dataset_copy.copy()

dataset_modified.drop(columns=["pH", "residual sugar"], inplace=True)

X_modified = dataset_modified.iloc[:, :-2]
y_modified = dataset_modified["quality"] > 6

X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(X_modified, y_modified, stratify=y_modified, random_state=42)

mlp_clf_exp2 = MLPClassifier(hidden_layer_sizes=[10, 15, 20], activation="relu", max_iter=10000, random_state=42)
pipeline_exp2 = make_pipeline(StandardScaler(), mlp_clf_exp2)
pipeline_exp2.fit(X_train_mod, y_train_mod)

accuracy_train_exp2 = pipeline_exp2.score(X_train_mod, y_train_mod)
accuracy_test_exp2 = pipeline_exp2.score(X_test_mod, y_test_mod)
print(f"\nAfter dropping features and different configurations:")
print(f"Training accuracy: {accuracy_train_exp2}, Test accuracy: {accuracy_test_exp2}")