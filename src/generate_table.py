import pandas as pd
import os
import config

def generate_latex_tables():
    results_base_dir = "../results"
    output_tex_file = os.path.join(results_base_dir, "summary_tables.tex")

    os.makedirs(results_base_dir, exist_ok=True)

    with open(output_tex_file, "w") as f:
        f.write("% Generated Summary Tables\n\n")

        for dataset in config.DATASETS:
            all_model_summaries = []

            for model in config.MODELS:
                csv_path = os.path.join(results_base_dir, model, dataset, f"{model}_{dataset}_results.csv")

                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        summary = df[['iou', 'dice', 'acc']].mean()

                        model_row = {
                            "Model": model,
                            "IoU": summary['iou'],
                            "Dice": summary['dice'],
                            "Accuracy": summary['acc']
                        }

                        all_model_summaries.append(model_row)
                    except Exception as e:
                        print(f"Error processing {csv_path}: {e}")

            if not all_model_summaries:
                print(f"No results found for dataset: {dataset}")
                continue

            df_results = pd.DataFrame(all_model_summaries)

            latex_table = df_results.to_latex(
                index=False,
                caption=f"Mean Performance Metrics for {dataset}",
                label=f"tab:{dataset.lower()}_mean",
                position="htbp",
                column_format="lccc",
                escape=False,
                float_format="{:0.4f}".format
            )

            if "\\centering" not in latex_table:
                latex_table = latex_table.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering")

            f.write(f"%% Table for {dataset}\n")
            f.write(latex_table)
            f.write("\n\n")

    print(f"Successfully saved all tables to {output_tex_file}")

if __name__ == '__main__':
    generate_latex_tables()