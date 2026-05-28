import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import os

RANDOM_STATE = 42

def run_lasso_ridge(df, drop_cols, plot_file_path):
    '''Run LASSO for feature selection on brain regions, then Ridge regression with bootstrapped CIs on selected features.'''
    X = df.drop(columns=drop_cols) # features are brain regions only
    y = df["Outcome"].values
    covariates = ["Age", "Sex", "No_Leads", "Target", "Base_MDSUPDRS"]

    X_full = df.drop(columns=covariates+drop_cols)
    y_full = df["Outcome"].values

    print(X_full.columns)

    n_repeats = 1000
    selection_counts = np.zeros(X_full.shape[1])

    n_selected_list = []   

    for i in range(n_repeats):

        X_res, y_res = resample(
        X_full, y_full,
        replace=True,
        random_state=RANDOM_STATE + i
    )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(
                alphas=np.logspace(-2, 2, 100),
                cv=3,
                max_iter=100000,
                random_state=RANDOM_STATE + i,
                tol=1e-2
            ))
        ])

        model.fit(X_res, y_res)

        coef = model.named_steps["lasso"].coef_
        selected_mask = np.abs(coef) > 1e-5

        selection_counts += selected_mask.astype(int)

        n_selected = np.sum(selected_mask)
        n_selected_list.append(n_selected)


    # plot selection stability
    selection_freq = selection_counts / n_repeats
    selection_df = pd.DataFrame({
        "Feature": X_full.columns,  
        "Selection_Frequency": selection_freq
    })
    summarize_results_df = selection_df.sort_values(
        by="Selection_Frequency", ascending=False
    ).reset_index(drop=True)


    # plot selection frequency
    plt.figure(figsize=(12, 6))
    plt.bar(summarize_results_df["Feature"], summarize_results_df["Selection_Frequency"])
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)   
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Selection Frequency", fontsize=14) 
    plt.title("Stability Selection Frequency of Lasso Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

    threshold = 0.3 # selection frequency threshold
    
    my_color = sns.color_palette("Set2", 10) # pull color from sns color palette
    plt.figure(figsize=(18, 18))

    plt.barh(
             summarize_results_df["Feature"], 
             summarize_results_df["Selection_Frequency"], 
             color=my_color[0], 
             edgecolor = "black", 
             linewidth=0.7, 
             height=1
             )
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2)
    
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)   
    plt.margins(y=0)
    plt.ylabel("Feature", fontsize=26)
    plt.xlabel("Selection Frequency", fontsize=26) 
    plt.title("Stability Selection Frequency of Lasso Features", fontsize=28)
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

    selected_features = selection_df[selection_df["Selection_Frequency"] > threshold]["Feature"].tolist()
    print(f"Selected features with selection frequency > {threshold}: {selected_features}")

    # selected brain region features and covarites for ridge regression
    selected_features = selected_features + ['Sex', 'Target', 
                                             'No_Leads', 'Age', "Base_MDSUPDRS"]

    X_sel= df[selected_features].copy()

    ridge = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=3) # ridge regression with 3-fold cross validation

    ridge.fit(StandardScaler().fit_transform(X_sel), y) # fit ridge regression

    # Bootstrap CIs
    n_boot = 1000
    boot_coefs = np.zeros((n_boot, len(selected_features)))

    for i in range(n_boot):

        X_res, y_res = resample(X_sel, y, random_state=i)

        X_res_scaled = StandardScaler().fit_transform(X_res)

        ridge.fit(X_res_scaled, y_res)

        boot_coefs[i] = ridge.coef_

    coef_median = boot_coefs.median(axis=0)
    ci_low = np.percentile(boot_coefs, 2.5, axis=0)
    ci_high = np.percentile(boot_coefs, 97.5, axis=0)


    results = pd.DataFrame({
        "Feature": selected_features,
        "Coefficient": coef_median,
        "Lower CI": ci_low,
        "Upper CI": ci_high,
    })

    results["Significant"] = (
        (results["Lower CI"] > 0) | (results["Upper CI"] < 0)
    ) 

    # ridge regression with bootstrap CI  plot
    x = np.arange(len(results))

    plt.figure(figsize=(14, 8))

    for i in range(len(results)):

        coef = results["Coefficient"].iloc[i]
        lower = results["Lower CI"].iloc[i]
        upper = results["Upper CI"].iloc[i]

        yerr = [[coef - lower], [upper - coef]]

        color = "blue" if results["Significant"].iloc[i] else "black"

        plt.errorbar(
                    x[i],
                    coef,
                    yerr=yerr,
                    fmt="o",
                    color=color,
                    capsize=5,
                    elinewidth=3
                )


    plt.axhline(0, linestyle="--", linewidth=1)

    label_map = {
        "Sex_factorized": "Sex",
        "Target_factorized": "Target",
        "No_Leads": "No Leads",
        "Base_MDSUPDRS": "Base MDSUPDRS"
    }

    labels = [
        label_map.get(f, f)
        for f in results["Feature"]
    ]

    plt.xticks(x, labels, rotation=45, fontsize=25, ha="right")
    plt.yticks(fontsize=25)
    plt.ylim(-2.25, 2.25)

    plt.xlabel("Features", fontsize=28)
    plt.ylabel("Effect on MoCA SDS", fontsize=28)

    plt.tight_layout()
    plt.savefig("FIXED_moca_ridge_coefficients_ci.png", dpi=300)
    plt.close()

    return [results.sort_values(
        by="Coefficient",
        key=np.abs,
        ascending=False
    )
    , selected_features, boot_coefs]


def main():
    # read in excel
    print("Reading in data...")

    # read in dataset
    moca_df = pd.read_excel("FIXED_cognitive_dataset.xlsx")

    # covariates to exclude from zero column check
    exclude_cols = ["Age", "Sex", "Target", "No_Leads", "SDS", "Base_MDSUPDRS"]

    # find zero columns excluding exclude_cols
    zero_cols = [col for col in moca_df.columns 
                if col not in exclude_cols and (moca_df[col] == 0).all()]

    # drop columns with all zeros (except covariates)
    moca_df.drop(columns=zero_cols, inplace=True)
    moca_df = moca_df.dropna().reset_index(drop=True)

    print(f"Number of patients in final dataset: {moca_df.shape[0]}")

    region_cols = [col for col in moca_df.columns if col not in exclude_cols]
    covariates = ["Age", "Sex", "Target", "No_Leads", "Base_MDSUPDRS"]

    moca_df = moca_df.rename(columns={"SDS":"Outcome"})
    drop_cols = ["Patient_ID", "Outcome"]

    # run lasso feature selection and ridge regression
    results, selected_features, boot_coefs = run_lasso_ridge(moca_df, drop_cols, "moca_lasso_stability_selection_frequency.png")
    print("Selected features from LASSO:", selected_features)
    results.to_csv("moca_lasso_ridge_selected_regions.csv", index=False)
    print(results)


if __name__ == "__main__":
    main()



