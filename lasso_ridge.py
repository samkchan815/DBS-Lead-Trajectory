import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import statsmodels.api as sm
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import os

RANDOM_STATE = 42

def parse_atlas_labels(label_file):
    '''Parse the atlas label file to create a mapping of label numbers to brain region names.'''
    label_dict = {} # initialize dictionary
    line_num= 0 # label numbers start at 1
    delimiter = '('

    # allow passing either a basename or full path
    if not os.path.isabs(label_file):
        label_file = os.path.join("atlas/", label_file)

    with open(label_file, 'r') as f: # open file as read only
        for line in f: # loop through each line in label text file
            line=line.replace(')', '') # remove closing parenthesis
            parts = line.strip().split(delimiter) # split by delimiter

            # save to dictionary in mini list
            # 0: full label name 1: abbreviation
            label_dict[line_num] = [parts[0].strip(), parts[1].strip()] # add to dict
            line_num += 1 # track label number (line # = label #s)
    return label_dict

def map_columns_to_atlas(col_names, atlas_dict):
    '''Map brain region names to brain atlas labels.'''
    region_names = []
    for col in col_names:
        # extract number from 'Region_1', 'Region_23', etc
        try:
            idx = int(col.split("_")[1]) - 1  # subtract 1 b/c atlas dict is starts at 0 (Region_0)
            region_names.append(atlas_dict[idx][1])  # abbreviated name
        except:
            region_names.append(col)  # fallback for non-region columns
    return region_names


def load_data(vectors_xlsx_list, outcomes_xlsx, atlas_file):
    '''Load dataset from excel files.'''
    # load & concatenate vector matrices
    vector_dfs = [pd.read_excel(xlsx) for xlsx in vectors_xlsx_list]
    vectors = pd.concat(vector_dfs, axis=0, ignore_index=True)

    # load outcomes from spreadsheet
    outcomes = pd.read_excel(outcomes_xlsx)
    outcomes = outcomes[
        ["PTID_Retro_Clin", "delta_ledd", "sex", "No_Leads", "age", "Target"]
    ]
    outcomes.columns = ["Patient_ID", "Outcome", "Sex", "No_Leads", "Age", "Target"]

    # merge
    df = vectors.merge(outcomes, on="Patient_ID", how="left")
    df = df.dropna().reset_index(drop=True)

    # map atlas labels
    atlas_dict = parse_atlas_labels(atlas_file)
    region_cols = [col for col in df.columns if col.startswith("Region_")]
    mapped_names = map_columns_to_atlas(region_cols, atlas_dict)
    df = df.rename(columns=dict(zip(region_cols, mapped_names)))

    return df


def run_lasso_ridge(df, drop_cols, plot_file_path):
    '''Run LASSO for feature selection on brain regions, then Ridge regression with bootstrapped CIs on selected features.'''
    X = df.drop(columns=drop_cols) # features are brain regions only
    y = df["Outcome"].values
    covariates = ["Age", "Sex", "No Leads", "Target", "Base MDSUPDRS"]

    X_full = df.drop(columns=covariates+drop_cols)
    y_full = df["Outcome"].values

    # get selected features above threshold
    threshold = 0.7 # theshold for whole
    # threshold = 0.5 # threshold for target stratified

    n_repeats = 1000 # number of bootstraps
    selection_counts = np.zeros(X_full.shape[1])

    n_selected_list = []   

    for i in range(n_repeats):

        X_res, y_res = resample(
            X_full, y_full,
            replace=True,
            random_state=RANDOM_STATE + i
            )

        model = Pipeline([
            ("scaler", StandardScaler()), # normalize
            ("lasso", LassoCV( # LASSO feature selection with 3-fold cross validation
                alphas=np.logspace(-2, 2, 100),
                cv=3,
                max_iter=100000,
                random_state=RANDOM_STATE + i,
                tol=1e-2
            ))
        ])

        model.fit(X_res, y_res)

        coef = model.named_steps["lasso"].coef_
        selected_mask = np.abs(coef) > 1e-5 # extract selected regions

        selection_counts += selected_mask.astype(int)

        n_selected = np.sum(selected_mask)
        n_selected_list.append(n_selected)

    # plot selection stability
    selection_freq = selection_counts / n_repeats
    selection_df = pd.DataFrame({
        "Feature": X_full.columns,  
        "Selection_Frequency": selection_freq
    })
    # sort values
    summarize_results_df = selection_df.sort_values(
        by="Selection_Frequency", ascending=False
    ).reset_index(drop=True)

    # take top half brain region features to plot
    top_features = summarize_results_df.head(len(summarize_results_df)//2)
    my_color = sns.color_palette("Set2", 10)
    plt.figure(figsize=(14, 12))

    plt.barh(
             top_features["Feature"], 
             top_features["Selection_Frequency"], 
             color=my_color[0], 
             edgecolor = "black", 
             linewidth=0.7, 
             height=1
             )
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2)
    
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)   
    plt.margins(y=0)
    plt.ylabel("Feature", fontsize=28)
    plt.xlabel("Selection Frequency", fontsize=28) 
    plt.title("Stability Selection Frequency of Lasso Features", fontsize=30)
    plt.tight_layout()
    plt.savefig(plot_file_path, dpi=300)
    plt.close()

    selected_features = selection_df[selection_df["Selection_Frequency"] > threshold]["Feature"].tolist()
    print(f"Selected features with selection frequency > {threshold}: {selected_features}")

    # ridge regression
    selected_features = selected_features + covariates # add covariates back in for ridge regression

    X_sel= df[selected_features].copy()

    # ridge regression with 3-fold cross validation
    ridge = RidgeCV(
        alphas=np.logspace(-4, 4, 100), 
                    cv=3
                    )

    # standardization
    ridge.fit(StandardScaler().fit_transform(X_sel), y)

    # Bootstrap CIs
    n_boot = 1000
    boot_coefs = np.zeros((n_boot, len(selected_features)))

    for i in range(n_boot):

        X_res, y_res = resample(X_sel, y, random_state=i)

        X_res_scaled = StandardScaler().fit_transform(X_res)

        ridge.fit(X_res_scaled, y_res)

        boot_coefs[i] = ridge.coef_

    coef_median = np.median(boot_coefs, axis=0)
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

    plt.figure(figsize=(10, 8))

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

    labels = [
        "Sex" if f == "Sex_factorized" else f
        for f in results["Feature"]
    ]

    plt.xticks(x, labels, rotation=45, fontsize=25, ha="right")
    plt.yticks(fontsize=25)
    plt.ylim(-50, 50)

    plt.xlabel("Features", fontsize=28)
    plt.ylabel("Effect on ΔLEDD (%)", fontsize=28)

    plt.tight_layout()
    plt.savefig("FIXED_ridge_coefficients_ci_04172026.png", dpi=300)
    plt.close()

    return [results, selected_features, boot_coefs]

def ols(df, selected_features):
    print("\nRunning OLS on selected LASSO features...")
    covariates = ["Age", "Sex", "No Leads", "Target", "Base MDSUPDRS"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=False)
    my_color = sns.color_palette("Set2", 10)

    # Selected Features + Covariates model
    X_full = df[selected_features + covariates]
    y = df["Outcome"]

    X_full = sm.add_constant(X_full)

    model_full = sm.OLS(y, X_full).fit()

    results_full = pd.DataFrame({
        "Feature": model_full.params.index,
        "Beta": model_full.params.values,
        "p_value": model_full.pvalues.values,
        "CI_low": model_full.conf_int()[0].values,
        "CI_high": model_full.conf_int()[1].values
    })

    print("\nFULL MODEL")
    print(model_full.summary())

    ax = axes[0]

    ax.barh(
        results_full["Feature"],
        results_full["Beta"],
        color=[
            my_color[0] if p < 0.05 else "grey"
            for p in results_full["p_value"]
        ],
        edgecolor="black",
        linewidth=0.7
    )

    ax.axvline(0, linestyle="--", color="grey")
    ax.tick_params(axis='both', labelsize=25)
    ax.set_title("Clinical + Imaging Features", fontsize=30)
    ax.set_xlabel("OLS Coefficient", fontsize=28)
    ax.set_ylabel("Feature", fontsize=28)
    ax.set_xlim(-50, 50)

    # covariates only model
    X_cov = df[covariates]

    X_cov = sm.add_constant(X_cov)

    model_cov = sm.OLS(y, X_cov).fit()

    results_cov = pd.DataFrame({
        "Feature": model_cov.params.index,
        "Beta": model_cov.params.values,
        "p_value": model_cov.pvalues.values,
        "CI_low": model_cov.conf_int()[0].values,
        "CI_high": model_cov.conf_int()[1].values
    })

    print("\nCOVARIATE ONLY MODEL")
    print(model_cov.summary())

    ax = axes[1]

    ax.barh(
        results_cov["Feature"],
        results_cov["Beta"],
        color=[
            my_color[0] if p < 0.05 else "grey"
            for p in results_cov["p_value"]
        ],
        edgecolor="black",
        linewidth=0.7
    )

    ax.axvline(0, linestyle="--", color="grey")
    ax.tick_params(axis='both', labelsize=25)
    ax.set_title("Clinical Covariates Only", fontsize=30)
    ax.set_xlabel("OLS Coefficient", fontsize=28)
    ax.set_ylabel("Feature", fontsize=28)
    ax.set_xlim(-50, 50)

    plt.tight_layout()
    plt.savefig("FIXED_ols_coefficients_ci.png", dpi=300, bbox_inches="tight")
    plt.close()
    return results


def stratified_lasso_ridge(df, drop_cols):

    results_dict = {}

    for target_val, label in zip([0, 1], ["GPi", "STN"]):

        print(f"\nRunning {label} analysis...")

        df_sub = df[df["Target"] == target_val].copy()

        # drop target column since we're stratifying by it
        # df_sub = df_sub.drop(columns=["Target"])

        results, selected_features, boot_coefs = run_lasso_ridge(df_sub, drop_cols, "lasso_stability_selection_frequency_" + label + ".png")

        # results_dict[label] = results
        results_dict[label] = {
            "results": results,
            "boot_coefs": boot_coefs,
            "features": selected_features
        }
        
        # save bootstrap coefficients for later plotting
        boot_df = pd.DataFrame(boot_coefs, columns=selected_features)
        boot_df.to_csv(f"bootstrap_coefficients_{label}.csv", index=False)

    print("\nFinished stratified analyses. Saved Results :))")

    # merge results to plot in single plot
    stn = results_dict["STN"]["results"].copy()
    gpi = results_dict["GPi"]["results"].copy()

    stn = stn.rename(columns={
        "Coefficient": "STN_coef",
        "Lower CI": "STN_low",
        "Upper CI": "STN_high"
    })

    gpi = gpi.rename(columns={
        "Coefficient": "GPi_coef",
        "Lower CI": "GPi_low",
        "Upper CI": "GPi_high"
    })

    merged = pd.merge(
        stn[["Feature", "STN_coef", "STN_low", "STN_high"]],
        gpi[["Feature", "GPi_coef", "GPi_low", "GPi_high"]],
        on="Feature",
        how="outer"
    ).fillna(0)

    merged.to_excel("stn_vs_gpi_lasso_ridge_results.xlsx", index=False)
    boot_df = make_bootstrap_df(results_dict)
    plot_side_by_side_bootstrap_scatter(boot_df, merged)

    # plot both stn and gpi in single plot
    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(14, 8))

    # STN
    plt.errorbar(
        x - width/2,
        merged["STN_coef"],
        yerr=[
            merged["STN_coef"] - merged["STN_low"],
            merged["STN_high"] - merged["STN_coef"]
        ],
        fmt='o',
        capsize=5,
        label="STN",
        color = 'blue'
    )

    # GPi
    plt.errorbar(
        x + width/2,
        merged["GPi_coef"],
        yerr=[
            merged["GPi_coef"] - merged["GPi_low"],
            merged["GPi_high"] - merged["GPi_coef"]
        ],
        fmt='o',
        capsize=5,
        label="GPi",
        color='green'
    )

    plt.axhline(0, linestyle="--")
    plt.yticks(fontsize=24)
    plt.xticks(x, merged["Feature"], rotation=45, ha="right", fontsize=24)
    plt.xlabel("Features", fontsize=26)
    plt.ylabel("Effect on ΔLEDD (%)", fontsize=26)
    plt.title("STN vs GPi Ridge Coefficients with Bootstrapped CIs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("stn_vs_gpi_coefficients.png", dpi=300)
    plt.close()

    return merged

def make_bootstrap_df(results_dict):
    dfs = []

    for label in ["STN", "GPi"]:
        res = results_dict[label]["results"]
        boots = results_dict[label]["boot_coefs"]
        features = res["Feature"].values

        for i, feature in enumerate(features):
            df_temp = pd.DataFrame({
                "Feature": feature,
                "Coefficient": boots[:, i],
                "Target": label
            })
            dfs.append(df_temp)

    return pd.concat(dfs, ignore_index=True)


def plot_side_by_side_bootstrap_scatter(boot_df, merged):

    features = merged["Feature"].tolist()
    x_positions = np.arange(len(features))

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)

    # STN plot
    stn_df = boot_df[boot_df["Target"] == "STN"]

    for i, feature in enumerate(features):
        vals = stn_df[stn_df["Feature"] == feature]["Coefficient"].values

        # jitter around x position
        jitter = np.random.normal(loc=0, scale=0.05, size=len(vals))

        axes[0].scatter(
            np.full_like(vals, x_positions[i]) + jitter,
            vals,
            alpha=0.05, 
            s=10,
            facecolors='grey', edgecolors='none',
        )

        # overlay median + CI
        row = merged.iloc[i]
        axes[0].errorbar(
            x_positions[i],
            row["STN_coef"],
            yerr=[
                [row["STN_coef"] - row["STN_low"]],
                [row["STN_high"] - row["STN_coef"]]
            ],
            fmt='o',
            color='blue',
            capsize=4
        )

    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("STN")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(features, rotation=45, ha="right")

    # GPi plot
    gpi_df = boot_df[boot_df["Target"] == "GPi"]

    for i, feature in enumerate(features):
        vals = gpi_df[gpi_df["Feature"] == feature]["Coefficient"].values

        jitter = np.random.normal(loc=0, scale=0.05, size=len(vals))

        axes[1].scatter(
            np.full_like(vals, x_positions[i]) + jitter,
            vals,
            alpha=0.05,
            s=10,
            facecolors='grey', edgecolors='none',
        )

        row = merged.iloc[i]
        axes[1].errorbar(
            x_positions[i],
            row["GPi_coef"],
            yerr=[
                [row["GPi_coef"] - row["GPi_low"]],
                [row["GPi_high"] - row["GPi_coef"]]
            ],
            fmt='o',
            color='green',
            capsize=4
        )

    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("GPi")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(features, rotation=45, ha="right")


    fig.suptitle("Bootstrap Coefficient Distributions by Target", fontsize=16)
    axes[0].set_ylabel("Coefficient (Effect on ΔLEDD)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("bootstrap_scatter_side_by_side.png", dpi=300)
    plt.close()


def main():
    # read in excel dataset
    print("Reading in data...")
    df = pd.read_excel("ledd_dataset.xlsx") # with atlas labels mapped
    print("Finished reading in data!\n")

    # clean data
    print("Cleaning data...")
    df["No_Leads_factorized"] = df["No_Leads"].map({"Bi":2, "Uni":1, "L":1, "R":1}) 
    df['Sex_factorized'], uniques = pd.factorize(df['Sex'])
    # display the mapping
    sex_mapping = dict(zip(uniques, range(len(uniques))))
    print("Sex Mapping:", sex_mapping)
    # factorize the 'Target' column
    df['Target_factorized'], uniques = pd.factorize(df['Target'])
    # display the mapping
    target_mapping = dict(zip(uniques, range(len(uniques))))
    print("Target Mapping:", target_mapping)

    df.drop(columns=["Sex", "Target", "No_Leads"], inplace=True) # drop old columns
    # columns to never dropped
    exclude_cols = ["Age", "Sex_factorized", "Target_factorized", "No_Leads_factorized", "Outcome", "Baseline_MDSUPDRS"]

    # find zero columns excluding those
    zero_cols = [col for col in df.columns 
                if col not in exclude_cols and (df[col] == 0).all()]

    df.drop(columns=zero_cols, inplace=True)
    df = df.rename(columns = {"Sex_factorized":"Sex", "Target_factorized":"Target", "No_Leads_factorized":"No Leads", "Baseline_MDSUPDRS": "Base MDSUPDRS"})

    df = df.dropna().reset_index(drop=True) # drop na

    drop_cols = ["Patient_ID", "Outcome"]

    results, selected_features, boot_coefs = run_lasso_ridge(df, drop_cols, "lasso_stability_selection_frequency.png")

    results.to_csv("lasso_ridge_selected_regions.csv", index=False)

    ols_results = ols(df, selected_features)
    ols_results.to_csv("ols_results.csv", index=False)

    # uncomment to run stratified model -- change threshold to 0.5
    # stratified_results = stratified_lasso_ridge(df, drop_cols)
    # print(stratified_results)

if __name__ == "__main__":
    main()



