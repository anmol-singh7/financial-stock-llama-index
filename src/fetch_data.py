import os
import urllib.request as request
import zipfile

data_url = "https://github.com/anmol-singh7/financial-stock-data/raw/main/articles.zip"

def download_and_extract():

    root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    download_path = os.path.join(root_folder, "articles.zip")
    
    print("Downloading the file...")
    filename, headers = request.urlretrieve(
        url=data_url,
        filename=download_path
    )
    print(f"File downloaded: {filename}")

    print("Extracting the file...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(root_folder)
    print("Extraction completed.")

    # Remove the ZIP file
    os.remove(download_path)
    print("ZIP file removed.")

print("Starting the process...")
download_and_extract()
