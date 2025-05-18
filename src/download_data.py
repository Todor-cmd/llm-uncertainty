import os
import requests

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download from {url}. Status code: {response.status_code}")

def download_all():
    file_list = [
        "train_en.tsv",
        "dev_en.tsv",
        "dev_test_en.tsv",
        "test_en.tsv",
        "test_en_gold.tsv"
    ]

    base_url = "https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/raw/main/task2/data/subtask-2-english/"

    for file_name in file_list:
        url = base_url + file_name
        save_path = f"src/data/{file_name}"
        download_file(url, save_path)

if __name__ == "__main__":
    download_all()
