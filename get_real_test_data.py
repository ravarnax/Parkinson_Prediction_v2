import pandas as pd

def generate_real_test_set():
    print("Fetching the authentic UCI Oxford Parkinson's dataset...")
    
    # Official UCI Repository URL for the Parkinson's dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        # Download the actual clinical dataset
        df = pd.read_csv(url)
        
        # Extract a 20% hold-out test set (40 records)
        # Using a fixed random_state ensures you get the same test set if you run it again
        test_df = df.sample(n=40, random_state=42)
        
        # Save it to a CSV file for your Streamlit app
        filename = "real_parkinsons_batch_test.csv"
        test_df.to_csv(filename, index=False)
        
        print("-" * 40)
        print(f"Successfully created: {filename}")
        print("-" * 40)
        print(f"Total test records    : {len(test_df)}")
        print(f"Healthy patients      : {(test_df['status'] == 0).sum()} (Ground Truth)")
        print(f"Parkinson's patients  : {(test_df['status'] == 1).sum()} (Ground Truth)")
        print("-" * 40)
        print("Upload this file to your Streamlit Batch Processor to check real accuracy.")
        
    except Exception as e:
        print(f"Error fetching data: {e}\nCheck your internet connection.")

if __name__ == "__main__":
    generate_real_test_set()
