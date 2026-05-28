import pandas as pd
import numpy as np

def combine_regions_and_outcome_spreadsheets(df, outcome_df):
    '''
    Take in MRI region matrix and pull outcome, age, sex, target, and number of leads from separate 
    Excel sheets, then combine into a single dataframe. Combined based on patient ID.
    '''
    combined_df = df.merge(outcome_df, on="Patient_ID", how="left")
    return combined_df

def combine_ct_and_mri_spreadsheets(mri_df, ct_df):
    '''
    Pull all patients from ct_df and mri_df who have outcome data in cog_df, 
    then combine into a single dataframe.
    Input: mri dataframe that has brain regions, covariates, and outcomes; 
    ct dataframe that has regions, covariates, and outcomes
    '''
    combined_df = pd.concat([mri_df, ct_df], axis=0, ignore_index=True)

    return combined_df

def fill_missing_from_ledd(
    df,
    ledd_file_path,
):
    """
    Fill missing values in df using corresponding values from LEDD spreadsheet.
    Existing values in df are NEVER overwritten.

    Outcome priority:
    LEDDpost2_Delta → LEDDpost3_Delta → LEDDpost4_Delta
    """

    # Load LEDD spreadsheet
    ledd_df = pd.read_excel(ledd_file_path)

    # Rename / standardize columns
    ledd_df = ledd_df.rename(columns={
        "PTID_Retro_Clin": "Patient_ID",
        "Sex": "Sex_LEDD",
        "No_Leads": "No_Leads_LEDD",
        "Age_DBS_On": "Age_LEDD",
        "Target_L_R": "Target_LEDD",
        "Base_MDSUPDRS": "MDSUPDRSIIIpre_Percent_TOTAL_V"
    })

    # Columns needed from LEDD
    ledd_cols = [
        "Patient_ID",
        "LEDDpost2_Delta",
        "LEDDpost3_Delta",
        "LEDDpost4_Delta",
        "Sex_LEDD",
        "No_Leads_LEDD",
        "Age_LEDD",
        "Target_LEDD",
        "MDSUPDRSIIIpre_Percent_TOTAL_V"
    ]

    ledd_df = ledd_df[ledd_cols].drop_duplicates("Patient_ID")

    # Merge
    df = df.merge(
        ledd_df,
        on="Patient_ID",
        how="left",
        validate="one_to_one"
    )

    print(f"Number of patients after merging with LEDD data: {df.shape[0]}")
    print(f"Columns after merge: {df.columns.tolist()}")

    # ---- OUTCOME: priority fill ----
    before = df["Outcome"].isna().sum()

    df["Outcome"] = df["Outcome"].combine_first(
        df["LEDDpost2_Delta"]
    ).combine_first(
        df["LEDDpost3_Delta"]
    ).combine_first(
        df["LEDDpost4_Delta"]
    )

    after = df["Outcome"].isna().sum()
    print(f"Outcome: filled {before - after} values")

    # ---- OTHER COVARIATES ----
    covariate_map = {
        "Sex": "Sex_LEDD",
        "No_Leads": "No_Leads_LEDD",
        "Age": "Age_LEDD",
        "Target": "Target_LEDD",
        "Baseline_MDSUPDRS": "MDSUPDRSIIIpre_Percent_TOTAL_V"
    }

    for df_col, ledd_col in covariate_map.items():
        if df_col not in df.columns:
            df[df_col] = np.nan

        before = df[df_col].isna().sum()
        df[df_col] = df[df_col].combine_first(df[ledd_col])
        after = df[df_col].isna().sum()
        print(f"{df_col}: filled {before - after} values")

    # Drop helper columns
    df = df.drop(
        columns=[
            "LEDDpost2_Delta",
            "LEDDpost3_Delta",
            "LEDDpost4_Delta",
            "Sex_LEDD",
            "No_Leads_LEDD",
            "Age_LEDD",
            "Target_LEDD",
            "MDSUPDRSIIIpre_Percent_TOTAL_V"
        ]
    )


    return df


def main():

    # ############### EXTRACT ALL COGNITIVE DATA FROM MRI AND CT IMAGES ################
    mri_df = pd.read_excel("mri_region_matrix_with_names.xlsx") # mri data we currently have

    ct_moca_df = pd.read_excel("moca_dataset_master_excel.xlsx") # ct data we currently have
    #rename columns in ct_df to match mri_df
    ct_moca_df = ct_moca_df.rename(columns={"Outcome": "SDS"})
    print(f"Number of patients with CT data: {ct_moca_df.shape[0]}")

    cog_df = pd.read_excel("data/subjects_cts_cogoutcomes_correct_ages.xlsx") # all cog info for both ct & mri

    cog_df = cog_df[["id", "SDS", "sex", "lead_num", "age", "target"]] # pull only relevant columns from cog_df
    cog_df.columns = ["Patient_ID", "SDS", "Sex", "No_leads", "Age", "Target"]
    cog_mri_full_df = combine_regions_and_outcome_spreadsheets(mri_df, cog_df) # combined mri regions w/ outcome data
    cog_mri_full_df= cog_mri_full_df.dropna().reset_index(drop=True)
    print(f"Number of patients with MRI data: {cog_mri_full_df.shape[0]}")
    cog_mri_full_df.to_excel("mri_regions_with_cog_outcomes.xlsx", index=False) # convert to excel for checking

    # factorize the No_leads column
    cog_mri_full_df["No_Leads_factorized"] = cog_mri_full_df["No_leads"].map({"Bi":2, "Uni":1, "L":1, "R":1}) 

    # factorize sex column
    cog_mri_full_df['Sex_factorized'] = cog_mri_full_df['Sex'].map({"M":0, "F":1})

    # factorize the 'Target' column
    cog_mri_full_df['Target_factorized'] = cog_mri_full_df['Target'].map({"STN":0, "GPi":1})

    cog_mri_full_df.drop(columns=["Sex", "Target", "No_leads"], inplace=True)

    cog_mri_full_df = cog_mri_full_df.rename(columns={
    "Sex_factorized": "Sex",
    "Target_factorized": "Target",
    "No_Leads_factorized": "No_Leads"
    })

    cog_dataset = combine_ct_and_mri_spreadsheets(cog_mri_full_df, ct_moca_df)
    ledd_path = "data/Retro_Clin_1.28.26_SD.xlsx"
    master_df = pd.read_excel(ledd_path)
    master_df = master_df.rename(columns={
        "PTID_Retro_Clin": "Patient_ID",
        "Sex": "Sex_LEDD",
        "No_Leads": "No_Leads_LEDD",
        "Age_DBS_On": "Age_LEDD",
        "Target_L_R": "Target_LEDD",
        "MDSUPDRSIIIpre_Percent_TOTAL_V": "Base_MDSUPDRS"
    })

    cog_dataset = cog_dataset.merge(
            master_df[["Patient_ID", "Base_MDSUPDRS"]], 
            on="Patient_ID",
            how="left"  
        )

    cog_dataset.to_excel("cognitive_dataset.xlsx", index=False)
    # print size of cog_dataset
    print(f"Number of cognitive patients: {cog_dataset.shape[0]}")

    ##################### EXTRACT ALL LEDD DATA FROM MRI AND CT IMAGES ################
    ct_ledd_df = pd.read_excel("output/master_excel_final.xlsx") # read in full ct dataset with all covariates and outcomes
    mri_df = pd.read_excel("mri_region_matrix_with_names.xlsx") # mri data w/ brain regions only
    ledd_df = pd.read_excel("data/subjects_cts_leddoutcomes_updated.xlsx") # all ledd info for both ct & mri

    print(f"Number of CT patients with LEDD Outcomes: {ct_ledd_df.shape[0]}", )
   
    ledd_df = ledd_df[["PTID_Retro_Clin", "delta_ledd", "sex", "age", "No_Leads", "Target"]]
    ledd_df.columns = ["Patient_ID", "Outcome", "Sex", "Age", "No_Leads", "Target"]
    mri_ledd_full_df = combine_regions_and_outcome_spreadsheets(mri_df, ledd_df)
    
    mri_ledd_full_df.to_excel("mri_regions_with_ledd_outcomes.xlsx", index=False)

    ledd_dataset = combine_ct_and_mri_spreadsheets(mri_ledd_full_df, ct_ledd_df)
    print(f"Number of patients with LEDD data: {ledd_dataset.shape[0]}")
    ledd_path = "data/Retro_Clin_1.28.26_SD.xlsx"
    df = fill_missing_from_ledd(ledd_dataset, ledd_path)

    df = df.dropna().reset_index(drop=True) # drop missing values
    df.to_excel("ledd_dataset.xlsx", index=False)

    
if __name__ == "__main__":
    main()