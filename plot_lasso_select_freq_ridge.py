import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def plot_lasso_stability_path(df, drop_cols):

    X = df.drop(columns=drop_cols + ["Age", "Sex", "Target", "No Leads"]) # brain region features only
    y = df["Outcome"].values

    alphas = np.logspace(-2, 1, 100) # alpha window
    n_boot = 1000 # bootstrap number

    features = X.columns
    n_features = len(features)

    # selection frequency per alpha per feature
    selection = np.zeros((len(alphas), n_features))

    for b in range(n_boot):

        X_res, y_res = resample(X, y, random_state=42 + b)

        scaler = StandardScaler() # standardize
        X_res_scaled = scaler.fit_transform(X_res)

        for i, a in enumerate(alphas): # loop through alpha window

            model = Lasso(alpha=a, max_iter=100000, tol = 1e-2, random_state=42 + b)
            model.fit(X_res_scaled, y_res)

            selected = (np.abs(model.coef_) > 1e-5)

            selection[i] += selected.astype(int)

    # convert to frequency
    selection = selection / n_boot
    # average stability across alpha values
    feature_stability = selection.mean(axis=0)

    selected_mask = feature_stability > 0.85  # threshold

    selected_features = X.columns[selected_mask] # extract selected features

    print("Selected stable features:")
    print(selected_features)

    # get color
    palette = sns.color_palette("Set2", len(selected_features))
    color_map = dict(zip(selected_features, palette))

    plt.figure(figsize=(12, 8))
    # plot selection stability across alpha 
    for j, feature in enumerate(features):

        if feature in selected_features:

            plt.plot(
                alphas,
                selection[:, j],
                linewidth=2.5,
                alpha=0.9,
                label=feature,
                color=color_map[feature]
            )

        else:

            plt.plot(
                alphas,
                selection[:, j],
                linewidth=1,
                alpha=0.15,
                color="gray"
            )

    # key for selected features
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1, fontsize=12)

    plt.xscale("log")
    plt.xlabel("Alpha (Regularization Strength)", fontsize=28)
    plt.ylabel("Selection Frequency", fontsize=28)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.title("Bootstrap LASSO Stability Paths", fontsize=30)

    plt.tight_layout()
    plt.savefig("lasso_stability_path.png", dpi=300)
    plt.close()


def main():
    # read in excel
    print("Reading in data...")
    #df = pd.read_excel("output/master_excel_final.xlsx")
    df = pd.read_excel("FIXED_ledd_dataset.xlsx")
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

    df.drop(columns=["Sex", "Target", "No_Leads", "Baseline_MDSUPDRS"], inplace=True)

    # columns to never drop
    exclude_cols = ["Age", "Sex_factorized", "Target_factorized", "No_Leads_factorized", "Outcome", "Baseline_MDSUPDRS"]

    # find zero columns excluding those
    zero_cols = [col for col in df.columns 
                if col not in exclude_cols and (df[col] == 0).all()]

    df.drop(columns=zero_cols, inplace=True) # drop any all 0 columns
    df = df.rename(columns = {"Sex_factorized":"Sex", "Target_factorized":"Target", "No_Leads_factorized":"No Leads", "Baseline_MDSUPDRS": "Base MDSUPDRS"})

    df = df.dropna().reset_index(drop=True) # drop  na

    drop_cols = ["Patient_ID", "Outcome"]

    plot_lasso_stability_path(df, drop_cols)

if __name__ == "__main__":
    main()



