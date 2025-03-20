import pandas as pd
import os

DATA_PATH = "../data/"

def read_xpt(file_path):
    """ Reads an .XPT file and returns a DataFrame. """
    return pd.read_sas(file_path, format="xport", encoding="utf-8")

datasets = {
    "demographics": read_xpt(os.path.join(DATA_PATH, "demographics/DEMO_J.xpt")),

    # Examination data
    "body_measures": read_xpt(os.path.join(DATA_PATH, "examination/BMX_J.xpt")),
    "blood_pressure": read_xpt(os.path.join(DATA_PATH, "examination/BPX_J.xpt")),
    "liver_ultrasound": read_xpt(os.path.join(DATA_PATH, "examination/LUX_J.xpt")),

    # Laboratory data
    "albumin_creatinine": read_xpt(os.path.join(DATA_PATH, "laboratory/ALB_CR_J.xpt")),
    "biochemistry": read_xpt(os.path.join(DATA_PATH, "laboratory/BIOPRO_J.xpt")),
    "complete_blood_count": read_xpt(os.path.join(DATA_PATH, "laboratory/CBC_J.xpt")),
    "glycohemoglobin": read_xpt(os.path.join(DATA_PATH, "laboratory/GHB_J.xpt")),
    "glucose": read_xpt(os.path.join(DATA_PATH, "laboratory/GLU_J.xpt")),
    "cholesterol_HDL": read_xpt(os.path.join(DATA_PATH, "laboratory/HDL_J.xpt")),
    "c_reactive_protein": read_xpt(os.path.join(DATA_PATH, "laboratory/HSCRP_J.xpt")),
    "insulin": read_xpt(os.path.join(DATA_PATH, "laboratory/INS_J.xpt")),
    "cholesterol_total": read_xpt(os.path.join(DATA_PATH, "laboratory/TCHOL_J.xpt")),
    "cholesterol_triglycerides": read_xpt(os.path.join(DATA_PATH, "laboratory/TRIGLY_J.xpt")),

    # Questionnaire data
    "alcohol": read_xpt(os.path.join(DATA_PATH, "questionnaire/ALQ_J.xpt")),
    "physical_activity": read_xpt(os.path.join(DATA_PATH, "questionnaire/PAQ_J.xpt")),
    "smoking": read_xpt(os.path.join(DATA_PATH, "questionnaire/SMQ_J.xpt")),
}

def merge_datasets(datasets):
    merged_df = datasets["demographics"]
    for name, df in datasets.items():
        if name != "demographics":  # We already have demographics
            merged_df = merged_df.merge(df, on="SEQN", how="outer")  # Keep all participants
    return merged_df

if __name__ == "__main__":
    print("Merging datasets with outer join...")
    merged_df = merge_datasets(datasets)
    print(f"Merged dataset shape: {merged_df.shape}")

# Check missing values percentage
missing_values = merged_df.isnull().sum() / len(merged_df) * 100
missing_values = missing_values.sort_values(ascending=False)
missing_values.to_csv("../reports/missing_values.csv")
print("Missing values report saved to ../reports/missing_values.csv")

# Step 1: Drop columns with more than 80% missing values
columns_to_drop = missing_values[missing_values > 80].index
merged_df.drop(columns=columns_to_drop, inplace=True)

# Step 2: Fill missing values (for numerical biomarkers)
for col in merged_df.select_dtypes(include=['float64', 'int64']).columns:
    merged_df[col].fillna(merged_df[col].median(), inplace=True)  # Fill missing values with median

# Save cleaned dataset to CSV
merged_df.to_csv("../reports/cleaned_dataset.csv", index=False)

print(f"Cleaned data's new shape: {merged_df.shape}")
print("Cleaned dataset saved to ../reports/cleaned_dataset.csv")

# List of columns to keep
selected_features = [
    # Demographics
    "RIDAGEYR", "RIAGENDR",

    # Body measurements
    "BMXWT", "BMXHT", "BMXBMI", "BMXLEG", "BMXARML", "BMXARMC", "BMXWAIST", "BMXHIP",

    # Blood pressure
    "BPXSY1", "BPXDI1",

    # Biomarkers
    "LBXGLU", "LBXIN", "LBXHSCRP", "LBXSCR", "LBXSGB", "LBXHGB", "LBXHCT", "LBXTC", "LBDLDL", "LBDHDD",

    # White blood cell count
    "LBXWBCSI",

    # Lifestyle
    "ALQ130", "PAQ605", "SMQ020"
]

final_df = merged_df[selected_features]
print(f"Feature selection done. Dataset shape before extra modifications: {final_df.shape}")

# Extra modifications after I was running the tests while training the model
# The two dataset versions below were used for the stacked model
extra_cols_to_drop = ["LUAPNME", "LUATECH", "SMDUPCA", "SMD100BR",
                      "RIAGENDR", "BMXHT", "BMXARML", "BMXBMI", "LBXHSCRP", "SIAPROXY"]

final_df = final_df.drop(columns=extra_cols_to_drop, errors="ignore")
final_df.to_csv("../reports/final_dataset.csv", index=False)
print(f"Final dataset shape after modifications: {final_df.shape}")
print("Final dataset saved to ../reports/final_dataset.csv")

