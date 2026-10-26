from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
med_map = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
            "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
            "troglitazone", "tolazamide", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]

# medicines given to fewer than 100 patients in train
rare_med_cols = ["glimepiride-pioglitazone", "metformin-pioglitazone", "acetohexamide", "metformin-rosiglitazone",
                 "troglitazone", "glipizide-metformin", "tolbutamide", "miglitol", "tolazamide", "chlorpropamide"]

onehot_cols = ["race", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
               "admission_type_id", "discharge_disposition_id", "admission_source_id"]

numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]

ordinal_cols = ["age", "max_glu_serum", "A1Cresult"] + [c for c in med_cols if c not in rare_med_cols]

binary_cols = ["gender", "change", "diabetesMed"]


def encode_labs(df):
    df = df.copy()
    df["max_glu_serum"] = df["max_glu_serum"].map(glu_map)   # replace each string by its number
    df["A1Cresult"] = df["A1Cresult"].map(a1c_map)
    return df


# apply the map to every medication column
def encode_meds(df):
    df = df.copy()
    df[med_cols] = df[med_cols].apply(lambda col: col.map(med_map))
    return df


# `diag_1/2/3` hold ~700 different codes. Making them categorize into 10 categories - ref paper from Dataset
def icd9_group(code):
    if code == "Unknown":
        return "Unknown"
    if code.startswith(("V", "E")):
        return "Other"
    n = float(code)
    if 250 <= n < 251:  return "Diabetes"
    if 390 <= n <= 459 or n == 785: return "Circulatory"
    if 460 <= n <= 519 or n == 786: return "Respiratory"
    if 520 <= n <= 579 or n == 787: return "Digestive"
    if 580 <= n <= 629 or n == 788: return "Genitourinary"
    if 140 <= n <= 239: return "Neoplasms"
    if 710 <= n <= 739: return "Musculoskeletal"
    if 800 <= n <= 999: return "Injury"
    return "Other"


def group_diagnoses(df):
    df = df.copy()
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].astype(str).map(icd9_group)
    return df


# text -> numbers, then one-hot the categories and scale the numbers
def build_preprocessor():
    prep = Pipeline([
        ("labs", FunctionTransformer(encode_labs)),
        ("meds", FunctionTransformer(encode_meds)),
        ("diags", FunctionTransformer(group_diagnoses)),
    ])
    encode = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(min_frequency=100, handle_unknown="infrequent_if_exist", sparse_output=False), onehot_cols),
            ("scale", StandardScaler(), numeric_cols + ordinal_cols),
            ("binary", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pre = Pipeline([("prep", prep), ("encode", encode)])
    pre.set_output(transform="pandas")
    return pre
