# Let's create dataset loaders and integration for multiple diabetes datasets
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Create sample data for the three main datasets since we can't directly download from external sources
# We'll create representative samples based on the research specifications

def create_pima_dataset():
    """Create Pima Indian Diabetes Dataset sample"""
    np.random.seed(42)
    n_samples = 768
    
    # Create features based on Pima dataset specifications
    data = {
        'Pregnancies': np.random.poisson(3, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 20, n_samples),
        'SkinThickness': np.random.normal(20, 15, n_samples),
        'Insulin': np.random.exponential(80, n_samples),
        'BMI': np.random.normal(32, 8, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
        'Age': np.random.normal(33, 12, n_samples)
    }
    
    # Apply realistic constraints
    data['Pregnancies'] = np.clip(data['Pregnancies'], 0, 17)
    data['Glucose'] = np.clip(data['Glucose'], 44, 199)
    data['BloodPressure'] = np.clip(data['BloodPressure'], 24, 122)
    data['SkinThickness'] = np.clip(data['SkinThickness'], 7, 99)
    data['Insulin'] = np.clip(data['Insulin'], 14, 846)
    data['BMI'] = np.clip(data['BMI'], 18.2, 67.1)
    data['DiabetesPedigreeFunction'] = np.clip(data['DiabetesPedigreeFunction'], 0.078, 2.42)
    data['Age'] = np.clip(data['Age'], 21, 81).astype(int)
    
    df = pd.DataFrame(data)
    
    # Create target variable with realistic correlations
    risk_score = (
        0.3 * (df['Glucose'] > 140) +
        0.25 * (df['BMI'] > 30) +
        0.2 * (df['Age'] > 40) +
        0.1 * (df['BloodPressure'] > 80) +
        0.1 * (df['DiabetesPedigreeFunction'] > 0.5) +
        0.05 * np.random.random(n_samples)
    )
    
    df['Outcome'] = (risk_score > 0.5).astype(int)
    
    # Add some 0 values for missing data (realistic for Pima dataset)
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in zero_features:
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_indices, feature] = 0
    
    print(f"Pima Dataset Created: {df.shape}")
    print(f"Diabetes cases: {df['Outcome'].sum()}, Non-diabetes: {(1-df['Outcome']).sum()}")
    
    return df

def create_frankfurt_dataset():
    """Create Frankfurt Hospital Germany Diabetes Dataset sample"""
    np.random.seed(123)
    n_samples = 2000
    
    # Create features based on Frankfurt dataset specifications (similar to Pima but larger)
    data = {
        'Pregnancies': np.random.poisson(2.5, n_samples),
        'Glucose': np.random.normal(115, 35, n_samples),
        'BloodPressure': np.random.normal(72, 18, n_samples),
        'SkinThickness': np.random.normal(22, 12, n_samples),
        'Insulin': np.random.exponential(75, n_samples),
        'BMI': np.random.normal(31, 7, n_samples),
        'DiabetesPedigreeFunction': np.random.exponential(0.45, n_samples),
        'Age': np.random.normal(35, 15, n_samples)
    }
    
    # Apply realistic constraints
    data['Pregnancies'] = np.clip(data['Pregnancies'], 0, 15)
    data['Glucose'] = np.clip(data['Glucose'], 40, 220)
    data['BloodPressure'] = np.clip(data['BloodPressure'], 30, 140)
    data['SkinThickness'] = np.clip(data['SkinThickness'], 5, 80)
    data['Insulin'] = np.clip(data['Insulin'], 10, 800)
    data['BMI'] = np.clip(data['BMI'], 16, 70)
    data['DiabetesPedigreeFunction'] = np.clip(data['DiabetesPedigreeFunction'], 0.05, 3.0)
    data['Age'] = np.clip(data['Age'], 18, 85).astype(int)
    
    df = pd.DataFrame(data)
    
    # Create target with slightly different risk profile
    risk_score = (
        0.35 * (df['Glucose'] > 126) +
        0.25 * (df['BMI'] > 28) +
        0.2 * (df['Age'] > 45) +
        0.1 * (df['BloodPressure'] > 85) +
        0.1 * (df['DiabetesPedigreeFunction'] > 0.6)
    )
    
    df['Outcome'] = (risk_score > 0.45).astype(int)
    
    print(f"Frankfurt Dataset Created: {df.shape}")
    print(f"Diabetes cases: {df['Outcome'].sum()}, Non-diabetes: {(1-df['Outcome']).sum()}")
    
    return df

def create_iraqi_dataset():
    """Create Iraqi Patient Dataset for Diabetes (IPDD) sample"""
    np.random.seed(456)
    n_samples = 1000
    
    # Create features based on IPDD specifications (12 features)
    data = {
        'Gender': np.random.binomial(1, 0.565, n_samples),  # 0=female, 1=male
        'Age': np.random.normal(53.7, 8.9, n_samples),
        'FBS': np.random.normal(10.1, 5.1, n_samples),  # Fasting Blood Sugar
        'BUN': np.random.normal(5.2, 3.3, n_samples),   # Blood Urea Nitrogen
        'Cr': np.random.normal(69.3, 62.3, n_samples),  # Creatinine
        'Chol': np.random.normal(4.9, 2.0, n_samples),  # Cholesterol
        'TG': np.random.normal(2.4, 1.4, n_samples),    # Triglycerides
        'BMI': np.random.normal(29.4, 4.9, n_samples),
        'LDL': np.random.normal(2.6, 1.1, n_samples),   # Low Density Lipoprotein
        'VLDL': np.random.normal(1.9, 3.7, n_samples),  # Very Low Density Lipoprotein
        'HDL': np.random.normal(1.2, 0.7, n_samples),   # High Density Lipoprotein
        'HbA1C': np.random.normal(8.3, 2.5, n_samples)  # Glycated Hemoglobin
    }
    
    # Apply realistic constraints
    data['Age'] = np.clip(data['Age'], 20, 79).astype(int)
    data['FBS'] = np.clip(data['FBS'], 3, 25)  # mmol/L
    data['BUN'] = np.clip(data['BUN'], 1, 15)
    data['Cr'] = np.clip(data['Cr'], 20, 300)
    data['Chol'] = np.clip(data['Chol'], 2, 12)
    data['TG'] = np.clip(data['TG'], 0.5, 8)
    data['BMI'] = np.clip(data['BMI'], 18, 50)
    data['LDL'] = np.clip(data['LDL'], 0.5, 8)
    data['VLDL'] = np.clip(data['VLDL'], 0.1, 3)
    data['HDL'] = np.clip(data['HDL'], 0.3, 3)
    data['HbA1C'] = np.clip(data['HbA1C'], 4, 18)
    
    df = pd.DataFrame(data)
    
    # Create 3-class target (0=non-diabetic, 1=pre-diabetic, 2=diabetic)
    # Based on HbA1C levels: <5.7 normal, 5.7-6.4 pre-diabetic, >6.4 diabetic
    conditions = [
        df['HbA1C'] < 5.7,
        (df['HbA1C'] >= 5.7) & (df['HbA1C'] < 6.5),
        df['HbA1C'] >= 6.5
    ]
    choices = [0, 1, 2]
    
    # Add some noise and other risk factors
    risk_adjustment = (
        0.1 * (df['FBS'] > 7) +
        0.1 * (df['BMI'] > 30) +
        0.1 * (df['Age'] > 50) +
        0.05 * (df['Chol'] > 5) +
        0.05 * np.random.random(n_samples)
    )
    
    base_class = np.select(conditions, choices, default=1)
    
    # Apply risk adjustment
    adjusted_prob = np.random.random(n_samples) + risk_adjustment
    final_class = base_class.copy()
    
    # Adjust some cases based on risk
    upgrade_mask = (adjusted_prob > 0.7) & (base_class < 2)
    final_class[upgrade_mask] = np.minimum(base_class[upgrade_mask] + 1, 2)
    
    df['Class'] = final_class
    
    print(f"Iraqi Dataset Created: {df.shape}")
    print(f"Class distribution - Non-diabetic: {(df['Class']==0).sum()}, Pre-diabetic: {(df['Class']==1).sum()}, Diabetic: {(df['Class']==2).sum()}")
    
    return df

# Create all datasets
pima_df = create_pima_dataset()
frankfurt_df = create_frankfurt_dataset()
iraqi_df = create_iraqi_dataset()

# Save datasets
pima_df.to_csv('pima_diabetes_dataset.csv', index=False)
frankfurt_df.to_csv('frankfurt_diabetes_dataset.csv', index=False)
iraqi_df.to_csv('iraqi_diabetes_dataset.csv', index=False)

print("\n=== DATASETS CREATED ===")
print(f"1. Pima Indian Diabetes Dataset: {pima_df.shape}")
print(f"2. Frankfurt Hospital Diabetes Dataset: {frankfurt_df.shape}")
print(f"3. Iraqi Patient Diabetes Dataset: {iraqi_df.shape}")

# Display sample data
print("\n=== PIMA DATASET SAMPLE ===")
print(pima_df.head())
print("\nFeatures:", pima_df.columns.tolist())

print("\n=== FRANKFURT DATASET SAMPLE ===")
print(frankfurt_df.head())
print("\nFeatures:", frankfurt_df.columns.tolist())

print("\n=== IRAQI DATASET SAMPLE ===")
print(iraqi_df.head())
print("\nFeatures:", iraqi_df.columns.tolist())