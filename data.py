import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import PowerTransformer
from sksurv.util import Surv
import re


def load_y_train() -> tuple[np.ndarray, np.ndarray]:
    """"
    Load the target data for training.
    """
    df_target = pd.read_csv("data/target_train.csv") # Target data (OS_YEARS and OS_STATUS)
    df_target = df_target.dropna() # Drop rows where 'OS_YEARS' is NaN if conversion caused any issues
    df_target['OS_YEARS'] = pd.to_numeric(df_target['OS_YEARS'], errors='coerce')
    y = Surv.from_dataframe(event='OS_STATUS', time='OS_YEARS', data=df_target) # Convert y to sksurv format
    
    # XGBoost needs a different format for y
    y_xgb = df_target["OS_YEARS"].values
    right_censored = df_target["OS_STATUS"] == 0
    # In XGBoost, negative values are considered right censored (cf. https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
    y_xgb[right_censored] *= -1  

    return y, y_xgb


def build_X(split, force_impute: bool=False) -> tuple[pd.DataFrame, pd.Series]:
    """"
    "Build the design matrix X and the patient IDs for the given split (train or test).
    """
    print(f"Loading {split} data...")
    # print(f"Applying transform: {transform}")

    # Load the data
    df_clinical = pd.read_csv(f"data/clinical_{split}.csv") # Clinical data (blood samples)
    df_molecular = pd.read_csv(f"data/molecular_{split}.csv") # Molecular data (genetic mutations)
    if split == 'train':
        # For training, we keep only the patients that are in the target data
        df_target = pd.read_csv("data/target_train.csv")
        df_target = df_target.dropna() # Drop rows where 'OS_YEARS' is NaN if conversion caused any issues
        kept_patients = df_target['ID']
        df_clinical = df_clinical[df_clinical['ID'].isin(kept_patients)]
        df_molecular = df_molecular[df_molecular['ID'].isin(kept_patients)]

    # Preprocess the data
    df_cli = preprocess_clinical(df_clinical) # extract clinical features
    df_mol = preprocess_molecular(df_molecular) # extract cytogenetic features
    df = df_cli.merge(df_mol, on='ID', how='left') # left merge because some patients have no molecular data associated

    if split == "test" or force_impute:
        # For test, we need to impute the values (while for train, we'll do it during cross-validation to avoid data leakage)
        float_imputer = SimpleImputer(strategy='most_frequent')
        float_cols = df_cli.select_dtypes(include=[np.float64]).columns # we only impute float columns for CLINICAL
        df[float_cols] = float_imputer.fit_transform(df[float_cols])
        df[df_mol.columns] = df[df_mol.columns].fillna(-1).astype(int) # for molecular, we fill missing values with -1 (because it's nonnegative integer data)
        
        # make sure there's no NaN values
        assert df.isna().sum().sum() == 0, "There are still NaN values in the dataframe"
    
        # if transform: # to standardize (since the data is very skewed for clinical)
        #     power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        #     df[float_cols] = power_transformer.fit_transform(df[float_cols])

        missing_cols = ['14',
                        '2KB_upstream_variant',
                        '3_prime_UTR_variant',
                        'complex_change_in_transcript',
                        'inframe_variant',
                        'initiator_codon_change',
                        'splice_site_variant',
                        'stop_retained_variant',
                        'synonymous_codon']
        # fill missing values with 0 for the columns that are not in the testing set
        for col in missing_cols:
            if col not in df.columns:
                print(f"Column {col} not found in the test set. Filling with 0.")
                # fill missing values with 0
                df[col] = 0
    patient_id = df['ID']
    X = df.drop(columns=['ID']) # drop ID column since it's not a feature
    return X, patient_id


def preprocess_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract clinical features from the clinical data.
    """
    df["CYTOGENETICS"] = df["CYTOGENETICS"].fillna("missing")
    df = add_cyto_patterns(df)
    df = more_cyto_features(df) # ADDED IN V2
    df = df.drop(columns=['CYTOGENETICS', 'CENTER']) # useless cols
    return df


def add_cyto_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Catch patterns in the CYTOGENETICS column."
    """
    # recurrent patterns in leukemia patients' cytogenetics
    patterns = {
        r'\-': 'CHR_LOSS',
        r'\+': 'CHR_GAIN',
        r'del': 'CHR_DELETION',
        r'der': 'CHR_DERIVATIVE',
        r'mar': 'CHR_MARKER',
        r'p\d+': 'CHR_SHORT_ARM',
        r'q\d+': 'CHR_LONG_ARM',
        r't\(': 'TRANSLOCATION',
        r'xx': 'CHR_XX',
        r'xy': 'CHR_XY',
        r'missing': 'MISSING_CYTOGENETICS',
    }
    for pattern, name in patterns.items():
        # df[name] = df['CYTOGENETICS'].str.contains(pattern, case=True, regex=True, na=False).astype(int)
        df[name] = df['CYTOGENETICS'].apply(lambda x: len(re.findall(pattern, x))) # sum instead of indicator
    return df


def more_cyto_features(df: pd.DataFrame) -> pd.DataFrame:
    df["n_+-"] = df["CYTOGENETICS"].str.count(r'\+|\-').fillna(0).astype(int) # count the number of + and - in the CYTOGENETICS column
    high_risk_markers = [r"-7", r"del\(5q\)", r"complex", r"t\(9;22\)"]
    for marker in high_risk_markers:
        df[marker] = df["CYTOGENETICS"].apply(lambda x: len(re.findall(marker, x))) # sum instead of indicator
    df["n_high_risk_markers"] = df[high_risk_markers].sum(axis=1) # sum the high risk markers
    df["cyto_complexity"] = df["CYTOGENETICS"].apply(lambda x: x.count(','))
    df["nlr"] = df["ANC"] / (df["WBC"] - df["ANC"]).replace(0, np.nan) # neutrophil to lymphocyte ratio
    df["plr"] = df["PLT"] / (df["WBC"] - df["ANC"]).replace(0, np.nan) # platelet to lymphocyte ratio
    df["blast_wbc_ratio"] = df["BM_BLAST"] / df["WBC"]
    df['ipss_r_score'] = df["BM_BLAST"] * 0.4 + df["HB"] * (-0.3) + df["PLT"] * (-0.2) + df["cyto_complexity"] * 0.5
    return df


def preprocess_molecular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cytogenetic features from the molecular data.
    """
    patient_groups = df.groupby("ID") # group data by patient ID (so each group contains all the mutations for a given patient)
    gene_stats = compute_gene_stats(patient_groups) # compute general statistics on the gene mutations
    key_genes = flag_key_genes(patient_groups) # flag presence of key genes
    counts = more_gene_features(patient_groups) # count the number of mutations per gene and chromosome # ADDED IN V2
    df = pd.concat([gene_stats, key_genes, counts], axis=1) # concatenate the two dataframes
    return df


def compute_gene_stats(patient_groups: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    """"
    Compute general statistics on the gene mutations."
    """
    # count the number of mutations per patient
    n_mut = patient_groups.size().rename("n_mut")
    # count the number of chromosomes affected
    n_chr = patient_groups["CHR"].nunique().rename("n_chr")
    # count the number of genes affected
    n_gene = patient_groups["GENE"].nunique().rename("n_gene")
    # count the number of effects
    n_effect = patient_groups["EFFECT"].nunique().rename("n_effect")
    # VAF statistics
    vaf_stats = patient_groups["VAF"].agg(["sum", "max"]).rename(
        columns={
            "sum": "vaf_sum",
            "max": "vaf_max",
        }
    )
    df_gene_stats = pd.concat([
        n_mut,
        n_chr,
        n_gene,
        n_effect,
        vaf_stats
    ], axis=1)
    return df_gene_stats


def flag_key_genes(patient_groups: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    # important genes for AML cf. https://pmc.ncbi.nlm.nih.gov/articles/PMC5767295/
    KEY_GENES = [
        'TP53', # added in v2
        'FLT3',
        'KIT',
        'RAS',
        'JAK2',
        'CBL',
        'TET2',
        'IDH1',
        'IDH2',
        'ASXL1',
        'WT1',
        'RUNX1',
        'DNMT3A',
        'MLL',
        'CEBPA',
        'NPM1',
    ]
    df_key_genes = pd.DataFrame(columns=KEY_GENES)
    for gene in KEY_GENES:
        # check if the gene is present in the mutations
        df_key_genes[gene] = patient_groups['GENE'].agg(lambda x: x.str.contains(gene, case=False, regex=False, na=False).any()) 
    df_key_genes = df_key_genes.astype(int) # convert to int
    return df_key_genes


def more_gene_features(patient_groups: pd.DataFrame) -> pd.DataFrame:
    """
    Add more features from the gene mutations.
    """
    effects_counts = patient_groups['EFFECT'].value_counts().unstack(fill_value=0)
    chr_counts = patient_groups['CHR'].value_counts().unstack(fill_value=0)
    df = pd.concat([effects_counts, chr_counts], axis=1)
    return df