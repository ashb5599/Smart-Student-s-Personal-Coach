import kagglehub
import shutil
import os

# Dictionary of Kaggle datasets and what we want to name them locally
datasets = {
    "lainguyn123/student-performance-factors": "StudentPerformanceFactors.csv",
    "ziya07/college-student-management-dataset": "Student_Management_Dataset.csv",
    "mdhossanr/computer-science-students-strength": "CS_Students_Strength.csv"
}

print("🚀 Starting dataset downloads...\n")

for kaggle_path, local_filename in datasets.items():
    print(f"⏳ Downloading {local_filename}...")
    
    # Download to cache
    path = kagglehub.dataset_download(kaggle_path)
    
    # Find the CSV file inside the downloaded cache folder
    for file in os.listdir(path):
        if file.endswith('.csv'):
            source_file = os.path.join(path, file)
            # Copy and rename it to our project folder
            shutil.copyfile(source_file, local_filename)
            print(f"✅ Successfully saved: {local_filename}\n")
            break # Move to the next dataset

print("🎉 All datasets are locked and loaded in your folder!")