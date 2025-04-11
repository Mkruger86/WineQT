import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score
from pathlib import Path

highQLabels = [False, True]

#Define the quality label for each wine
y_true = [True, False, True, True, False, True, True, False, True, True, False, True, False, False, False, True, True, False, True, True]
#Assume that the quality for each wine has been predicted as followsaA
y_pred = [False, False, True, True, False, True,False, False, True, True, False, True, True, False, False, True, True, False, True, False]

#Establish the confusion matrix
cm=confusion_matrix(y_true, y_pred, labels=highQLabels)
# Calculate and print the precision score: 
# TP/(FP+TP) = 9/(1+9) = 0.9
precision = precision_score(y_true, y_pred)
# Calculate and print the recall score: 
# TP/(FN+TP) = 9/(3+9) = 0.75
recall = recall_score(y_true, y_pred)
# Calculate and print the accuracy score: 
# (TN+TP)/(TN+FP+TP+FN) = (7+9)/(7+1+9+3) = 0.8
accuracy = accuracy_score(y_true, y_pred)

# Print the scores
print(precision)
print(recall)
print(accuracy)

# Save the scores to a CSV file only if it doesn't exist
results_csv_path = Path("WineQT_Initial_Scores.csv")
if not results_csv_path.exists():
    results = {
        "Metric": ["Precision", "Recall", "Accuracy"],
        "Score": [precision, recall, accuracy]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv_path, index=False)

#Show the matrix
CM_File = Path("WineQT_Initial_CM.png")
if not CM_File.exists():
    disp = ConfusionMatrixDisplay(cm, display_labels=highQLabels)
    disp.plot()
    plt.savefig(CM_File)

plt.show()

