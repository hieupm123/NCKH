from netCDF4 import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os

def read_nc_file(file_path, var_name):
    with Dataset(file_path, mode='r') as nc_file:
        data = np.array(nc_file.variables[var_name][:], dtype=np.float32)
        if data.ndim > 2:
            data = data.squeeze()  # Loại bỏ các chiều không cần thiết
        return data

def process_ir_image(file_paths, var_name, gammas, rights, lefts):
    channels = []
    for i, file_path in enumerate(file_paths):
        ir = read_nc_file(file_path, var_name)
        
        # Áp dụng gamma correction
        ir = ((ir / rights[i]) ** gammas[i] * rights[i]).astype(np.float32)
        
        # Chuẩn hóa về 0-255
        ir = np.clip(ir, lefts[i], rights[i])
        ir = ((ir - lefts[i]) / (rights[i] - lefts[i]) * 255).astype(np.uint8)
        ir = 255 - ir  # Đảo ngược màu
        
        channels.append(ir)
    
    # Đổi thứ tự kênh: IR3 -> Red, IR1 -> Green, IR2 -> Blue
    image = np.stack([channels[2], channels[0], channels[1]], axis=-1)
    return Image.fromarray(image, mode="RGB")

def process_excel_and_convert_images(excel_path, data_dir, output_dir, var_name):
    df = pd.read_csv(excel_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        storm_id = str(row['International number ID'])
        most_time = row['most recent time'].replace("-", "").replace(":", "").replace(" ", "")[:12]
        
        file_paths = [
            f"{data_dir}/{storm_id}/netcdf/{most_time}/conbaomini_{most_time}_IR1bt.nc",
            f"{data_dir}/{storm_id}/netcdf/{most_time}/conbaomini_{most_time}_IR2bt.nc",
            f"{data_dir}/{storm_id}/netcdf/{most_time}/conbaomini_{most_time}_IR3bt.nc"
        ]
        
        if all(os.path.exists(fp) for fp in file_paths):
            gammas = [1.8, 1.8, 1.3]
            rights = [290, 290, 260]
            lefts = [160, 160, 160]
            
            image = process_ir_image(file_paths, var_name, gammas, rights, lefts)
            output_path = f'{output_dir}/{storm_id}/image/{most_time}/conbaomini_{most_time}.png'
            image.save(output_path)
            print(f"Saved: {output_path}")
        else:
            print(f"Missing files for {most_time}")

# Định nghĩa đường dẫn
excel_path = "test.csv"  # Đường dẫn đến file Excel
data_dir = "/datausers3/kttv/tien/ClassificationProject/mtsat_data/working"
output_dir = "/datausers3/kttv/tien/ClassificationProject/mtsat_data/working"
var_name = "tbb"

# Xử lý dữ liệu từ Excel và chuyển đổi ảnh
process_excel_and_convert_images(excel_path, data_dir, output_dir, var_name)