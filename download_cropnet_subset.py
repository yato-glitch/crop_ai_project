import os
from cropnet.data_downloader import DataDownloader

def fetch_data_subset():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing official CropNet DataDownloader...")
    
    # Target specific FIPS codes and years supported by CropNet
    target_fips = ["17019", "17167", "19153", "31055"] # Champaign, Sangamon, Polk, Douglas
    target_years = ["2018", "2019", "2020", "2021", "2022"]
    
    print(f"Downloading data for FIPS: {target_fips} across years {target_years}...")
    
    # Initialize the actual library downloader class
    downloader = DataDownloader(target_dir=output_dir)
    
    try:
        # Download USDA yield data records for target counties
        downloader.download_USDA("Corn", fips_codes=target_fips, years=target_years)
        downloader.download_USDA("Soybeans", fips_codes=target_fips, years=target_years)
        
        # Download Sentinel-2 imagery subset (AG format)
        downloader.download_Sentinel2(fips_codes=target_fips, years=target_years, image_type="AG")
        
        print(f"Download complete! Files saved safely to {output_dir}")
    except Exception as e:
        print(f"API download encountered an issue (check internet or package limits): {e}")

if __name__ == "__main__":
    fetch_data_subset()