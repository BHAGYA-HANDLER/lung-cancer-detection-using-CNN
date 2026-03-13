from zipfile import ZipFile
#here we are extracting the dataset from the zip file to the current directory
zip_path = "dataset.zip"

with ZipFile(zip_path,'r') as zip:
    zip.extractall()
    
print("Dataset Extracted")