import pandas as pd
import numpy as np
import os
import time

def generate_clinical_load_test(filename="parkinsons_150mb_test.csv", target_size_mb=150):
    print(f"Initializing load test generation for {filename}...")
    start_time = time.time()

    # The exact 22 features required by your Streamlit app
    features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
        "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", 
        "spread1", "spread2", "D2", "PPE"
    ]

    # Approximate rows needed to hit ~150MB (roughly 250-280 bytes per row)
    # 600,000 rows is a safe target for ~150MB
    num_rows = 600_000 

    # Generate synthetic data using normal distributions around realistic clinical baselines
    np.random.seed(42)
    
    data = {
        # Optional columns to test your app's drop logic
        "name": [f"Patient_{i:07d}" for i in range(num_rows)],
        "status": np.random.choice([0, 1], size=num_rows, p=[0.75, 0.25]), # 25% PD risk
        
        # Fundamental Frequency
        "MDVP:Fo(Hz)": np.random.uniform(80.0, 260.0, num_rows),
        "MDVP:Fhi(Hz)": np.random.uniform(100.0, 600.0, num_rows),
        "MDVP:Flo(Hz)": np.random.uniform(60.0, 250.0, num_rows),
        
        # Jitter
        "MDVP:Jitter(%)": np.random.uniform(0.001, 0.03, num_rows),
        "MDVP:Jitter(Abs)": np.random.uniform(0.00001, 0.0003, num_rows),
        "MDVP:RAP": np.random.uniform(0.001, 0.02, num_rows),
        "MDVP:PPQ": np.random.uniform(0.001, 0.02, num_rows),
        "Jitter:DDP": np.random.uniform(0.003, 0.06, num_rows),
        
        # Shimmer
        "MDVP:Shimmer": np.random.uniform(0.009, 0.1, num_rows),
        "MDVP:Shimmer(dB)": np.random.uniform(0.08, 1.0, num_rows),
        "Shimmer:APQ3": np.random.uniform(0.004, 0.05, num_rows),
        "Shimmer:APQ5": np.random.uniform(0.005, 0.07, num_rows),
        "MDVP:APQ": np.random.uniform(0.007, 0.1, num_rows),
        "Shimmer:DDA": np.random.uniform(0.013, 0.15, num_rows),
        
        # Non-linear & Entropy
        "NHR": np.random.uniform(0.0001, 0.3, num_rows),
        "HNR": np.random.uniform(8.0, 33.0, num_rows),
        "RPDE": np.random.uniform(0.2, 0.7, num_rows),
        "DFA": np.random.uniform(0.5, 0.9, num_rows),
        "spread1": np.random.uniform(-8.0, -2.0, num_rows),
        "spread2": np.random.uniform(0.006, 0.5, num_rows),
        "D2": np.random.uniform(1.0, 4.0, num_rows),
        "PPE": np.random.uniform(0.04, 0.5, num_rows),
    }

    print("Compiling DataFrame...")
    df = pd.DataFrame(data)

    # Round to 5 decimal places to mimic actual hardware export precision and control file size
    for col in features:
        df[col] = df[col].round(5)

    print(f"Writing {num_rows:,} records to CSV...")
    df.to_csv(filename, index=False)
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    elapsed = time.time() - start_time
    
    print("-" * 40)
    print("TEST FILE GENERATED SUCCESSFULLY")
    print("-" * 40)
    print(f"File Name : {filename}")
    print(f"Total Rows: {num_rows:,}")
    print(f"File Size : {file_size_mb:.2f} MB")
    print(f"Time Taken: {elapsed:.2f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    generate_clinical_load_test()
