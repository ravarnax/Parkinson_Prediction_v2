import os
import requests

def download_parkinsons_data():
    # The DIRECT raw link to the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    # Define the path
    save_path = os.path.join('data', 'raw', 'parkinsons.data')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Downloading data from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for connection errors
        
        # Write the content to the file
        with open(save_path, 'wb') as f:
            f.write(response.content)
            
        print(f"✅ Success! Data saved to: {save_path}")
        
        # Verify it's not HTML
        with open(save_path, 'r') as f:
            first_line = f.readline()
            if "<!DOCTYPE html>" in first_line or "<html" in first_line:
                print("❌ ERROR: The file is still an HTML page. The link might be blocked.")
            else:
                print(f"👀 First line check: {first_line.strip()} (Looks like a valid CSV header!)")
                
    except Exception as e:
        print(f"❌ Failed to download: {e}")

if __name__ == "__main__":
    download_parkinsons_data()