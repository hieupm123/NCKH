#! /usr/bin/python
import sys, os, datetime
import shutil
from netCDF4 import Dataset       # Read / Write NetCDF4 files
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt  # Plotting library
import matplotlib.patches as patches
from numpy import *
import re

# import pandas with shortcut 'pd' 
import pandas as pd   
import dateutil.parser as parser
import numpy as np

def download_geoss(year,month,day,hour,minute,tcid,out_dir,bands,lon_lat,store, c_map = None, replace = True):
    # hàm này tiến hành tải dữ liệu trên web về, số kênh tải về phụ thuộc vào giá trị band
    bands_list = bands.split(",")
    # working/tcid/image/yearmonthdayhourminute/biendong_....
    netcdf_dir = os.path.join(out_dir, str(tcid) + "/netcdf")
    png_dir = os.path.join(out_dir, str(tcid) + "/image")
    temp_dir = os.path.join(out_dir, str(tcid) + "/temp")
    os.makedirs(netcdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    for band in bands_list:
        
        script_path = "/home/future/Subject/ComputerVision/NCKH/count2tbb.sh"
        cmd = f"bash {script_path} {year} {month} {day} {hour} {minute} {str(tcid)} {lon_lat[0]} {lon_lat[1]} {lon_lat[2]} {lon_lat[3]}"
        
        print(f"Đang chạy: {cmd}")
        exit_code = os.system(cmd)

        if exit_code == 0:
            print(f"count2tbb.sh chạy thành công!")
            return 1
        else:
            print(f"Lỗi khi chạy count2tbb.sh! Mã lỗi: {exit_code}")
            return 0
        
def upload_to_remote(local_dir, remote_dir, name_file):
    ssh_command = f'ssh -p 7722 -o ControlPath=/tmp/ssh_control tien@hpc01.vast.vn "mkdir -p {remote_dir}"'
    scp_command = f"scp -o ControlPath=/tmp/ssh_control -P 7722 -r {local_dir} tien@hpc01.vast.vn:{remote_dir}"
    unzip_command = f'ssh -p 7722 -o ControlPath=/tmp/ssh_control tien@hpc01.vast.vn "unzip {remote_dir}/{name_file} -d {remote_dir}"'
    print(f"Đảm bảo thư mục tồn tại: {ssh_command}")
    
    ret1 = os.system(ssh_command)
    
    if ret1 == 0:
        ret2 = os.system(scp_command)
        if ret2 == 0:
            print("Upload successful!")
            os.system(unzip_command)
            rm_command = f'ssh -p 7722 -o ControlPath=/tmp/ssh_control tien@hpc01.vast.vn "rm -r {remote_dir}/{name_file}"'
            os.system(rm_command)
            return True
        else:
            print(f"Upload failed with exit code: {ret2}")
            return False
    else:
        print(f"Failed to create remote directory with exit code: {ret1}")
        return False

def process_row(typhoon_period):
    bands = typhoon_period['Bands and number']
    tcid = typhoon_period["International number ID"]
    date = typhoon_period['most recent time']
  
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    hour = date.strftime("%H")
    minute = date.strftime("%M")
    lon_lat = (typhoon_period['min_lon'], typhoon_period['max_lon'], typhoon_period['min_lat'], typhoon_period['max_lat'])
    
    print(f"Processing ID: {tcid} | Date: {year}-{month}-{day} {hour}:{minute} | Bands: {bands}")
    status = download_geoss(year, month, day, hour, minute, tcid, OUT_DIR, bands, lon_lat, store=False, c_map='Greys')
    
    # Nếu tải về thành công, tiến hành upload folder của tcid đến remote
    if status == 1:
        ######################################################
        folder_path = f'{OUT_DIR}/{tcid}/netcdf/{year}{month}{day}{hour}{minute}'
        zip_path = f"{folder_path}.zip"
        
        # Nén các file con bên trong folder_path mà không bao gồm folder_path chính
        # Dùng lệnh 'cd' để chuyển đến folder và nén tất cả file con (dùng "*" để bao gồm tất cả)
        zip_command = f"cd {folder_path} && zip -r {zip_path} *"
        print(f"Nén các file trong {folder_path} thành file {zip_path} ...")
        zip_status = os.system(zip_command)
        ########################################################
        adding = f'/{str(tcid)}/netcdf/{year}{month}{day}{hour}{minute}/*'
        add = f'/{str(tcid)}/netcdf/{year}{month}{day}{hour}{minute}'
        local_folder = f'{OUT_DIR}{adding}'
        local_folder_to_delete = f'{OUT_DIR}{add}'
        # Trên remote, folder "mtsat_data" đã tồn tại; upload folder con của tcid vào đó
        remote_folder = f"/datausers3/kttv/tien/ClassificationProject/mtsat_test/working/{str(tcid)}/netcdf/{year}{month}{day}{hour}{minute}"
        print(f"Uploading folder {local_folder} to remote folder {remote_folder} ...")
        upload_status = upload_to_remote(zip_path, remote_folder, f'{year}{month}{day}{hour}{minute}.zip')
        # upload_status = upload_to_remote(local_folder, remote_folder)
        if upload_status:
            print(f"Upload thành công cho TCID {tcid}. Đang xóa folder {local_folder}...")
            os.system(f"rm -r {local_folder_to_delete}")
            os.system(f"rm -r {zip_path}")
        else:
            print(f"Upload thất bại cho TCID {tcid}.")
    else:
        print(f"Tải về thất bại cho TCID {tcid}.")
    
    return (tcid, status)

def get_data_from_csv(path):
    data = pd.read_csv(path, 
                       parse_dates=['most recent time'],
                       usecols=['International number ID', 'Name of the storm', 'Time of analysis',
                                'Grade', 'Latitude of the center', 'Longitude of the center',
                                'Central pressure', 'Maximum sustained wind speed',
                                'The longest radius of 50kt winds or greater', 
                                'Bands and number', 'min_lon', 'max_lon', 'min_lat', 'max_lat', 'most recent time'])
    
    max_threads = 10
    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_tcid = {executor.submit(process_row, data.iloc[i, :]): data.iloc[i, :]["International number ID"] for i in range(len(data))}

        for future in as_completed(future_to_tcid):
            tcid = future_to_tcid[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"TCID {tcid} gặp lỗi: {e}")

    print("Tất cả tiến trình đã hoàn thành! Kết quả:")
    for tcid, status in results:
        print(f"TCID {tcid}: {'Tải thành công' if status == 1 else 'Đã tồn tại hoặc lỗi'}")

    
BIN_DIR = "/home/future/Subject/ComputerVision/NCKH"
# OUT_DIR = "/home/tc/himawari-download/Cycle_Typhoon_Project/himawari_dataset/working"
OUT_DIR =  "/home/future/Subject/ComputerVision/NCKH/mtsat_test/working"
if __name__ == "__main__":
    BASE = "/home/future/Subject/ComputerVision/NCKH"
    path = "/test.csv"
    df = pd.read_csv(BASE + path, parse_dates=['Time of analysis'])
    # print(np.min(df.loc[:, "Latitude of the center"]), np.max(df.loc[:, "Latitude of the center"]))
    # print(np.min(df.loc[:, "Longitude of the center"]), np.max(df.loc[:, "Longitude of the center"]))
    file_exten = path.split(".")[1]
    if file_exten == "csv":
        get_data_from_csv(BASE + path)
        print("Ending-Successfull: 1 - ok, 0-error")
        # print("Saving Status: 1 - ok, 0 - error: ", lon_lat_to_pixel(BASE + path))
    else:
        print("Unknown file format, please add file format handling! ")