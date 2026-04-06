import pandas as pd
import numpy as np
import os

def generate_icu_data(num_records=15000):
    np.random.seed(42)
    
    # Generate age (mostly older adults, mean ~ 60)
    age = np.random.normal(loc=60, scale=15, size=num_records)
    age = np.clip(age, 18, 100).astype(int)
    
    # Base health factor (hidden variable to correlate features somewhat realistically)
    health_factor = np.random.uniform(0, 1, num_records)
    
    # Heart rate (normal ~ 60-100, sick > 100 or < 60)
    heart_rate = np.random.normal(loc=80, scale=15, size=num_records) + (health_factor * 20)
    
    # Blood pressure
    systolic_bp = np.random.normal(loc=120, scale=20, size=num_records) - (health_factor * 30)
    diastolic_bp = np.random.normal(loc=80, scale=10, size=num_records) - (health_factor * 15)
    
    # Respiratory rate (normal 12-20, sick > 20)
    respiratory_rate = np.random.normal(loc=16, scale=4, size=num_records) + (health_factor * 10)
    respiratory_rate = np.clip(respiratory_rate, 8, 40)
    
    # Oxygen saturation (SpO2, normal 95-100%, sick < 95%)
    oxygen_saturation = np.random.normal(loc=98, scale=2, size=num_records) - (health_factor * 10)
    oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
    
    # Temperature (normal ~ 36.5 - 37.5, sick can be lower or higher)
    temperature = np.random.normal(loc=37.0, scale=0.8, size=num_records) + (health_factor * np.random.choice([-1, 1], num_records) * 1.5)
    
    # Glucose level
    glucose_level = np.random.normal(loc=110, scale=30, size=num_records) + (health_factor * 50)
    
    # Glasgow Coma Scale (GCS) - added as standard ICU score, range 3 to 15
    gcs = np.random.normal(loc=14, scale=2, size=num_records) - (health_factor * 8)
    gcs = np.clip(gcs, 3, 15).astype(int)
    
    # Calculate probability of mortality (Logistic-like function based on anomalies)
    logit = -5.0 + \
            (age * 0.03) + \
            (abs(heart_rate - 80) * 0.02) + \
            (abs(120 - systolic_bp) * 0.02) + \
            (respiratory_rate * 0.05) + \
            ((100 - oxygen_saturation) * 0.1) + \
            (abs(temperature - 37.0) * 0.3) + \
            ((15 - gcs) * 0.4)
            
    # Add some noise
    logit += np.random.normal(0, 1, num_records)
    
    prob_mortality = 1 / (1 + np.exp(-logit))
    
    # Generate labels (approx 20-30% mortality is typical in severe ICU datasets like MIMIC)
    mortality_label = (np.random.rand(num_records) < prob_mortality).astype(int)
    
    # Introduce some missing values to test preprocessing
    for feature in [heart_rate, systolic_bp, respiratory_rate, oxygen_saturation, glucose_level]:
        missing_idx = np.random.choice(num_records, size=int(0.02 * num_records), replace=False)
        feature[missing_idx] = np.nan
        
    df = pd.DataFrame({
        'age': age,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'respiratory_rate': respiratory_rate,
        'oxygen_saturation': oxygen_saturation,
        'temperature': temperature,
        'glucose_level': glucose_level,
        'gcs_score': gcs,
        'mortality_label': mortality_label
    })
    
    # Add a few duplicate rows to satisfy preprocessing requirement
    df = pd.concat([df, df.sample(100)], ignore_index=True)
    
    # Ensure directory exists and save
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/ICU_DATASET.csv', index=False)
    print(f"Generated {len(df)} records. Saved to data/ICU_DATASET.csv")
    print(f"Mortality Rate: {df['mortality_label'].mean() * 100:.2f}%")

if __name__ == "__main__":
    generate_icu_data()
