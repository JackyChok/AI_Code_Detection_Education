import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv

class MetricsEvaluator:
    def __init__(self, filename, show_plot = False):
        self.show_plot = show_plot
        self.data = pd.read_csv(filename, encoding_errors='backslashreplace')

    def evaluate(self, column_name):
        pred_human_col_name = column_name + "_answer_human_binary"
        pred_GPT_col_name = column_name + "_answer_GPT_binary"

        pred_human = self.data[pred_human_col_name]
        pred_GPT = self.data[pred_GPT_col_name]

        actual_human = np.ones_like(pred_human)
        actual_GPT = np.zeros_like(pred_GPT)

        human = {
            "accuracy": accuracy_score(actual_human, pred_human),
            "precision": precision_score(actual_human, pred_human),
            "recall": recall_score(actual_human, pred_human, zero_division=1),
            "f1": f1_score(actual_human, pred_human),
        }

        GPT = {
            "accuracy": accuracy_score(actual_GPT, pred_GPT),
            "precision": precision_score(actual_GPT, pred_GPT),
            "recall": recall_score(actual_GPT, pred_GPT, zero_division=1),
            "f1": f1_score(actual_GPT, pred_GPT),
        }

        print("Human:", human)
        print("GPT:", GPT)

        return human, GPT

    def evaluate_with_auc(self, column_name):
        pred_human_col_name = column_name + "_answer_human_binary"
        pred_GPT_col_name = column_name + "_answer_GPT_binary"

        pred_human = self.data[pred_human_col_name]
        pred_GPT = self.data[pred_GPT_col_name]

        # Concatenate the two columns to create a new column
        pred_combined = np.concatenate([pred_human, pred_GPT])

        actual_human = np.ones_like(pred_human)
        actual_GPT = np.zeros_like(pred_GPT)

        # Concatenate the two columns to create a new column
        actual_combined = np.concatenate([actual_human, actual_GPT])
        combined_auc = roc_auc_score(actual_combined, pred_combined)

        combined = {
            "accuracy": accuracy_score(actual_combined, pred_combined),
            "precision": precision_score(actual_combined, pred_combined),
            "recall": recall_score(actual_combined, pred_combined, zero_division=1),
            "f1": f1_score(actual_combined, pred_combined),
            "auc": combined_auc
        }

        print("Combined:", combined)

        # Plot the ROC curve for the combined predictions
        fpr, tpr, thresholds = roc_curve(actual_combined, pred_combined)
        plt.plot(fpr, tpr)
        plt.title('ROC Curve (AUC = {:.2f})'.format(combined_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        print(tpr)
        print(fpr)

        plt.show()

        return combined, fpr, tpr
    
    def calculate(self, column_name, output_file):
        # Open the CSV file in write mode
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(['Metric', 'Value'])
            print("            Actual   ")
            print("       |   | 1  | 0  |")
            print("Predict| 1 | tp | fp |")
            print("       | 0 | fn | tn |")
            self.detailed_evaluation(column_name, writer)
            self.overall_evaluation(column_name, writer)

    def detailed_evaluation(self, column_name, writer):
        group_data = self.data.groupby(["variant","Source Name"])
        for key, group in group_data:
            self.metrics_performance(column_name, group, key, writer)
        group_data = self.data.groupby("variant")
        for key, variant_group in group_data:
            self.metrics_performance(column_name, variant_group, key, writer)

    def overall_evaluation(self, column_name, writer):
        self.metrics_performance(column_name, self.data, "Overall", writer)

    def metrics_performance(self, column_name, dataframe, label, writer):
        pred_human_col_name = column_name + "_answer_human_binary"
        pred_GPT_col_name = column_name + "_answer_GPT_binary"

        pred_human = dataframe[pred_human_col_name]
        pred_GPT = dataframe[pred_GPT_col_name]
        actual_human = np.ones_like(pred_human)
        actual_GPT = np.zeros_like(pred_GPT)

        actual = np.concatenate((actual_human, actual_GPT))
        predict = np.concatenate((pred_human, pred_GPT))
        combined_auc = roc_auc_score(actual, predict)
        cfm = confusion_matrix(actual, predict)
        tn, fp, fn, tp = cfm.ravel()
        print(f"\n\n=============== {label} Performance ===============")

        print("For human:")
        print("True Positive Rate  (TPR): ", tp/(tp+fn))
        print("False Negative Rate (FNR): ", fn/(tp+fn))
        print("For GPT:")
        print("True Negative Rate  (TNR): ", tn/(fp+tn))
        print("False Positive Rate (FPR): ", fp/(fp+tn))

        combined = {
            "accuracy": accuracy_score(actual, predict),
            "precision": precision_score(actual, predict),
            "recall": recall_score(actual, predict, zero_division=1),
            "f1": f1_score(actual, predict),
            "auc": combined_auc
        }
        print()
        for key, value in combined.items():
            print("{:<15}: {}".format(str(key).capitalize(), value))
        writer.writerow([f'=============== {label} Performance ==============='])
        writer.writerow(['For human:'])
        writer.writerow(['True Positive Rate (TPR):', tp/(tp+fn)])
        writer.writerow(['False Negative Rate (FNR):', fn/(tp+fn)])
        writer.writerow(['For GPT:'])
        writer.writerow(['True Negative Rate (TNR):', tn/(fp+tn)])
        writer.writerow(['False Positive Rate (FPR):', fp/(fp+tn)])
        writer.writerow(['Combined:'])
        writer.writerow(['Accuracy:', combined['accuracy']])
        writer.writerow(['Precision:', combined['precision']])
        writer.writerow(['Recall:', combined['recall']])
        writer.writerow(['F1:', combined['f1']])
        writer.writerow(['Auc:', combined['auc']])
        writer.writerow([])  # Empty row between results
        if label == "Overall" and self.show_plot:
            self.plot_auc(actual, predict, combined_auc)

    def plot_auc(self, actual, predict, combined_auc):
        # Plot the ROC curve for the combined predictions
        fpr, tpr, thresholds = roc_curve(actual, predict)
        plt.plot(fpr, tpr)
        plt.title('ROC Curve (AUC = {:.2f})'.format(combined_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # print("")
        # print(tpr)
        # print(fpr)
        plt.show()


if __name__ == "__main__":
    # Example usage
    filename = "variants_prompt.csv"
    CDM = "gpt2outputdetector"
    evaluator = MetricsEvaluator(filename)
    output_file = CDM+"_results.csv"
    # evaluator.evaluate_with_auc("Writer")
    
    evaluator.calculate(CDM, output_file)
