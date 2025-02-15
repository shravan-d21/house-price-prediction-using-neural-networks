import kagglehub

# Download latest version
path = kagglehub.dataset_download("aariyan101/usa-housingcsv")

print("Path to dataset files:", path)