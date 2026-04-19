import pandas as pd
import shutil
import os

# Paths
csv_path = "multichannel-glaucoma-benchmark-dataset/metadata - standardized.csv"
image_root = "multichannel-glaucoma-benchmark-dataset/full-fundus/full-fundus"

output_glaucoma = "dataset_new/glaucoma"
output_normal = "dataset_new/normal"

os.makedirs(output_glaucoma, exist_ok=True)
os.makedirs(output_normal, exist_ok=True)

# Load metadata
df = pd.read_csv(csv_path)

# Keep only labeled rows
df = df[df['cdr_avg'].notna()]

print("Labeled samples:", len(df))

count_g = 0
count_n = 0

for _, row in df.iterrows():
    name = row['names']
    cdr = row['cdr_avg']

    img_path = os.path.join(image_root, f"{name}.png")

    if not os.path.exists(img_path):
        continue

    # Label rule
    if cdr >= 0.6:
        shutil.copy(img_path, os.path.join(output_glaucoma, f"{name}.png"))
        count_g += 1
    else:
        shutil.copy(img_path, os.path.join(output_normal, f"{name}.png"))
        count_n += 1

print("Glaucoma images:", count_g)
print("Normal images:", count_n)
print("DONE")