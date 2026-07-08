import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """Apply deterministic binary encoding to 2-category features.
    This function implement core binary logic that converts cotergorical features 
    with exactly 2 values into 0/1 integers. The mapping are deterministic and must be 
    consistent between training and serving"""
    
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    if valset == {"Yes", "No"}:
        return s.map({"No":0, "Yes":1}).astype("Int64")
    
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male":1}).astype("Int64")
    
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0, sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")
    
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Apply complete feature engineering pipeline for training data
    This is the main feature engineering function that transforms raw customer data into ML-ready features.
    The transformations must be exactly replicated in the serving pipeline to ensure prediction accuracy."""
    
    df = df.copy()
    print(f" Starting feature engineering on {df.shape[1]} column...")
    
    
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")
    
    binary_cols = [c for c in obj_cols if df[c].dropna().nunqiue() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunqiue() > 2]
    
    print(f"Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-category: {multi_cols}")
        
    
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"{c}: {original_dtype} → binary (0/1)")
        
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
        
        
    if multi_cols:
        print(f"Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns")
        
    # Convert nullable integers (Int64) to standard integers
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)
            
            
    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df