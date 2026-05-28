# DBS Lead Trajectory

This project performs image registration and atlas-based feature extraction for deep brain stimulation (DBS) lead trajectory analysis in individuals with Parkinson's disease. The pipeline registers patient postoperative imaging (CT and MRI) to MNI space, transforms a brain atlas into patient space, and identifies atlas-defined brain regions intersected by DBS lead trajectory segmentations. The brain region features are used for downstream statistical modeling of postoperative motor and cognitive outcomes.

## Project Overview
The workflow consists of the following:
1. Registration of postoperative CT or MRI images to MNI space using ANTsPy
2. Transformation of an atlas from MNI space into patient space
3. Extraction of brain regions intersected by DBS lead trajectories
4. Construction of binary feature matrices representing lead-region intersections
5. Statistical modeling using stability-selected LASSO and ridge regression model
6. Visualization of lead trajectories, atlas overlaps, and regression results

## Code Files

### ```ct_registration.py```
Performs registration of postoperative CT images to MNI space, transforms the atlas into patient space, and extracts binary brain-region intersection vectors from DBS lead segmentations. Returns folder containing all transformed images and excel file containing brain region vectors.

### ```mri_registration.py```
MRI version of image registration pipeline. Returns folder containing all transformed images and excel file containing brain region vectors.

### ```combined_dataset.py```
Combine CT image and MR image datasets for downstream analysis and appends covariate data including age, sex, number of leads, DBS target, and baseline MDS-UPDRS score. Outputs LEDD outcomes dataset and MoCA outcomes dataset in excel file format.

### ```motor_lasso_ridge.py```
Performs stability-selected LASSO feature selection and ridge regression modeling for motor outcomes. Also runs OLS modesl comparing model containing clnical covariates only and model containing both clinical covariates and selected brain region features. Returns LASSO feature selection bar plot, ridge regression coefficients and 95% confidence intervals plot, and csv files containing OLS results and ridge regression results.

### ```cog_lasso_ridge.py```
Performs stability-selected LASSO feature selection and ridge regression modeling for cognitive outcomes. Returns LASSO feature selection bar plot, ridge regression coefficients and 95% confidence intervals plot, and csv files containing ridge regression results.

###  ```plot_lasso_path.py```
Generates plots of LASSO coefficient paths and feature selection frequencies across different values of alpha (regularization parameter).

### ```plot_strat_target.py```
Creates plot illustrating ridge regression results from dataset stratified by target (GPi vs. STN).

## Data

### LEDD_dataset
Motor outcome dataset containing:
- Patient ID
- Percent change in levodopa equivalent daily dose (ΔLEDD)
- Binary brain region features
- DBS target (STN vs. GPi)
- Baseline Movement Disorder Society Unified Parkinson’s Disease Rating Scale Part III (MDS-UPDRS) score
- Sex
- Age

### Cognitive_dataset
- Patient ID
- MoCA Simple Discrepency Score (SDS)
- Binary brain region features
- DBS target (STN vs. GPi)
- Baseline Movement Disorder Society Unified Parkinson’s Disease Rating Scale Part III (MDS-UPDRS) score
- Sex
- Age

## Atlases
Brain atlas folder containing atlas file and adjoining ROI text file for region identification.

# Licenses
Copyright 2026 UCSF
