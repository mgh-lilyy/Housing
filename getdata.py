import os
import shutil
import kagglehub

# Tải dataset (sẽ tải về thư mục mặc định của kagglehub)
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

# Thư mục đích muốn lưu dataset
destination_path = "G:/Sem5/Python/price/dataset"

# Di chuyển toàn bộ dataset đến thư mục đích
shutil.move(path, destination_path)

data = kagglehub.dataset_download("yasserh/housing-prices-dataset")
