import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_sdf_files(url, folder_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # If the response was unsuccessful, this will raise a HTTPError
    except requests.exceptions.HTTPError as errh:
        print("HTTP Error:", errh)
        return
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:", errc)
        return
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:", errt)
        return
    except requests.exceptions.RequestException as err:
        print("Something went wrong with the request:", err)
        return

    soup = BeautifulSoup(response.text, "html.parser")

    for link in tqdm(soup.find_all("a")):
        file_link = link.get("href")
        if file_link.endswith(".sdf.gz"):
            file_url = url + file_link
            file_path = os.path.join(folder_path, file_link)

            try:
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except requests.exceptions.HTTPError as errh:
                print("HTTP Error:", errh)
            except requests.exceptions.ConnectionError as errc:
                print("Error Connecting:", errc)
            except requests.exceptions.Timeout as errt:
                print("Timeout Error:", errt)
            except requests.exceptions.RequestException as err:
                print("Something went wrong with the request:", err)


url = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF/"
folder_path = (
    "----"  # replace with your folder path
)
download_sdf_files(url, folder_path)
