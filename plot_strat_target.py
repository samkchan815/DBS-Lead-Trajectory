import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_side_by_side_bootstrap_scatter(boot_df, merged):

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    my_colors = sns.color_palette("Paired") # get paired color palette from seaborn

    # STN plot
    stn_df = boot_df[boot_df["Target"] == "STN"]
        
    stn_features = merged[
        ~((merged["STN_coef"] == 0) & 
        (merged["STN_low"] == 0) & 
        (merged["STN_high"] == 0))
    ]["Feature"].values
    x_stn = np.arange(len(stn_features))
    
    for i, feature in enumerate(stn_features):
        vals = stn_df.loc[stn_df["Feature"] == feature, "Coefficient"].values

        jitter = np.random.normal(0, 0.05, size=len(vals))

        axes[0].scatter(
            np.full(len(vals), x_stn[i]) + jitter,
            vals,
            alpha=0.2,
            s=10,
            color=my_colors[0]
        )

        row = merged[merged["Feature"] == feature]
        if not row.empty:
            row = row.iloc[0]

            axes[0].errorbar(
                x_stn[i],
                row["STN_coef"],
                yerr=[
                    [row["STN_coef"] - row["STN_low"]],
                    [row["STN_high"] - row["STN_coef"]]
                ],
                fmt='o',
                color=my_colors[1],
                capsize=4,
                elinewidth=3
            )
    axes[0].set_ylim(-50, 50)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("STN", fontsize=28)
    axes[0].set_xticks(x_stn)
    axes[0].set_xticklabels(stn_features, rotation=45, ha="right", fontsize=25)

    # GPi plot
    gpi_df = boot_df[boot_df["Target"] == "GPi"]

    gpi_features = merged[
        ~((merged["GPi_coef"] == 0) & 
        (merged["GPi_low"] == 0) & 
        (merged["GPi_high"] == 0))
    ]["Feature"].values
    
    x_gpi = np.arange(len(gpi_features))

    for i, feature in enumerate(gpi_features):
        vals = gpi_df.loc[gpi_df["Feature"] == feature, "Coefficient"].values

        jitter = np.random.normal(0, 0.05, size=len(vals))

        axes[1].scatter(
            np.full(len(vals), x_gpi[i]) + jitter,
            vals,
            alpha=0.2,
            s=10,
            color=my_colors[2]
        )

        row = merged[merged["Feature"] == feature]
        if not row.empty:
            row = row.iloc[0]

            axes[1].errorbar(
                x_gpi[i],
                row["GPi_coef"],
                yerr=[
                    [row["GPi_coef"] - row["GPi_low"]],
                    [row["GPi_high"] - row["GPi_coef"]]
                ],
                fmt='o',
                color=my_colors[3],
                capsize=4,
                elinewidth=3
            )

    axes[1].axhline(0, linestyle="--")
    axes[1].set_ylim(-50, 50)
    axes[1].set_title("GPi", fontsize=28)
    axes[1].set_xticks(x_gpi)
    axes[1].set_xticklabels(gpi_features, rotation=45, ha="right", fontsize=25)

    # fig.suptitle("Bootstrap Coefficient Distributions by Target", fontsize=22)
    axes[0].set_ylabel("Effect on ΔLEDD (%)", fontsize=28)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=25)

    plt.xlabels = ["Features", "Features"]
    plt.ylabels = ["Effect on ΔLEDD (%)", "Effect on ΔLEDD (%)"]
    plt.tight_layout()
    plt.savefig("bootstrap_scatter_side_by_side.png", dpi=300)
    plt.close()

def wide_to_long(df, target_label):
    df_long = df.melt(var_name="Feature", value_name="Coefficient")
    df_long["Target"] = target_label
    return df_long


def main():
    gpi_results_path = "bootstrap_coefficients_GPi.csv"
    stn_results_path = "bootstrap_coefficients_STN.csv"
    results_summary_path = "stn_vs_gpi_lasso_ridge_results.xlsx"

    # read in results files
    gpi_boot_df = pd.read_csv(gpi_results_path)
    stn_boot_df = pd.read_csv(stn_results_path)
    merged = pd.read_excel(results_summary_path)

    # add target labels
    gpi_boot_df["Target"] = "GPi"
    stn_boot_df["Target"] = "STN"

    boot_df = pd.concat([gpi_boot_df, stn_boot_df], ignore_index=True)

    # convert to long format
    boot_df_long = boot_df.melt(
        id_vars="Target",
        var_name="Feature",
        value_name="Coefficient"
    )

    # remove Target feature rows
    boot_df_long = boot_df_long[boot_df_long["Feature"] != "Target"]

    plot_side_by_side_bootstrap_scatter(boot_df_long, merged)


if __name__ == "__main__":
    main()
