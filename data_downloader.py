from datasets import load_dataset

def download_wikitext(save_dir="."):
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    # Save the dataset to disk
    for split in dataset.keys():
        file_path = f"{save_dir}/wikitext-2-{split}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset[split]:
                f.write(item["text"] + "\n")

    print(f"WikiText-2 dataset has been downloaded and saved to {save_dir}")

# Example usage
download_wikitext(save_dir=".")


