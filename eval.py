# Libraries
import pandas as pd

def evaluate_results(models: list) -> pd.DataFrame:
    results = pd.DataFrame(
        columns=[
            "Model",
            "Accuracy (avg)",
            "Accuracy (var)",
            "Macro F1 Score (avg)",
            "Macro F1 Score (var)",
        ]
    )

    for model in models:
        result_df = pd.read_csv(f"results/result_df_{model}.csv", index_col=0)

        result = {
            "Model": model,
            "Accuracy (avg)": result_df.mean()["Accuracy"],
            "Accuracy (var)": result_df.var()["Accuracy"],
            "Macro F1 Score (avg)": result_df.mean()["Macro F1 Score"],
            "Macro F1 Score (var)": result_df.var()["Macro F1 Score"],
        }

        results = results.append(result, ignore_index=True)

    return results


# Table 1: Load and compare unsupervised results

results_unsupervised = evaluate_results(["ae", "dbscan", "if", "lof"])
print(results_unsupervised)


# Table 2: Load and compare supervised results
results_supervised = evaluate_results(["cnn", "encoder", "lstm", "rf"])
print(results_supervised)