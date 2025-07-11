import cv2
import numpy as np
import random
import math
import sys
import os
import pandas as pd
from datetime import datetime
import warnings
from datetime import timedelta

# --- Hàm trợ giúp (Giữ nguyên, không thay đổi và không in) ---

def latlon_to_pixel(lat, lon, min_lat, max_lat, min_lon, max_lon, img_height, img_width):
    """Chuyển đổi tọa độ Lat/Lon sang tọa độ pixel (hàng, cột). Raises ValueError on error."""
    try:
        lat, lon = float(lat), float(lon)
        min_lat, max_lat = float(min_lat), float(max_lat)
        min_lon, max_lon = float(min_lon), float(max_lon)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Lỗi chuyển đổi tọa độ sang số thực: {e}.")

    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
        raise ValueError(f"Tọa độ tâm ({lat}, {lon}) ngoài giới hạn ([{min_lat},{max_lat}], [{min_lon},{max_lon}]).")

    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    # Sử dụng ngưỡng nhỏ để tránh chia cho số gần 0
    if abs(lat_range) < 1e-9 or abs(lon_range) < 1e-9:
         raise ValueError(f"Khoảng lat ({lat_range}) hoặc lon ({lon_range}) quá nhỏ hoặc bằng 0.")
    pixel_x = ((lon - min_lon) / lon_range) * img_width
    pixel_y = ((max_lat - lat) / lat_range) * img_height
    col = int(round(pixel_x))
    row = int(round(pixel_y))
    col = max(0, min(col, img_width - 1))
    row = max(0, min(row, img_height - 1))
    return row, col

def rotate_image(image, angle_degrees, center=None, scale=1.0):
    """Quay ảnh một góc cho trước. Returns rotated image and rotation matrix."""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, scale)
    border_value = 0 # Nền đen
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)
    return rotated, M

def transform_point(point, M):
    """Áp dụng ma trận biến đổi affine lên một điểm. Returns (col_new, row_new)."""
    (x, y) = point # Nhận vào (col_orig, row_orig)
    vec = np.array([x, y, 1])
    transformed_vec = M @ vec
    return int(round(transformed_vec[0])), int(round(transformed_vec[1]))

def find_max_sum_crop(rotated_image, center_row_new, center_col_new, crop_size):
    """Tìm vùng crop chứa tâm có tổng pixel LỚN NHẤT. Returns (best_r, best_c). Raises ValueError on error."""
    (h, w) = rotated_image.shape[:2]
    crop_h, crop_w = crop_size, crop_size

    # Chuyển sang thang độ xám một cách an toàn
    if len(rotated_image.shape) == 3:
        if rotated_image.shape[2] == 4: gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGRA2GRAY)
        elif rotated_image.shape[2] == 3: gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        else: gray_rotated = rotated_image[:,:,0] # Fallback không an toàn lắm
    elif len(rotated_image.shape) == 2: gray_rotated = rotated_image.copy()
    else: raise ValueError(f"Định dạng ảnh không được hỗ trợ (số chiều: {len(rotated_image.shape)})")

    integral_image = cv2.integral(gray_rotated, sdepth=cv2.CV_64F)
    max_sum = -1.0
    best_crop_coords = None

    # Xác định phạm vi tìm kiếm
    min_r = max(0, center_row_new - (crop_h - 1))
    max_r = min(h - crop_h, center_row_new)
    min_c = max(0, center_col_new - (crop_w - 1))
    max_c = min(w - crop_w, center_col_new)

    # Kiểm tra xem có thể tìm kiếm không
    if min_r > max_r or min_c > max_c :
        # Cố gắng sử dụng tọa độ mặc định nếu không tìm được phạm vi
        default_r = max(0, min(center_row_new - crop_h // 2, h - crop_h))
        default_c = max(0, min(center_col_new - crop_w // 2, w - crop_w))
        if h >= crop_h and w >= crop_w and 0 <= default_r <= h - crop_h and 0 <= default_c <= w - crop_w:
             # Không in cảnh báo, chỉ trả về tọa độ mặc định
             return default_r, default_c
        else:
             raise ValueError(f"Ảnh ({h}x{w}) / tâm ({center_row_new},{center_col_new}) không cho phép cắt {crop_size}x{crop_size}.")

    # Duyệt tìm max sum
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            r2 = r + crop_h
            c2 = c + crop_w
            # Bỏ qua kiểm tra biên vì integral image đã có padding
            current_sum = integral_image[r2, c2] - integral_image[r, c2] - integral_image[r2, c] + integral_image[r, c]
            if current_sum > max_sum:
                max_sum = current_sum
                best_crop_coords = (r, c)

    if best_crop_coords is None:
         # Nếu không tìm thấy sau khi duyệt (dù đã có kiểm tra ban đầu), thì đó là lỗi
         # Có thể trả về tọa độ mặc định ở đây nếu muốn an toàn hơn
         default_r = max(0, min(center_row_new - crop_h // 2, h - crop_h))
         default_c = max(0, min(center_col_new - crop_w // 2, w - crop_w))
         if h >= crop_h and w >= crop_w and 0 <= default_r <= h - crop_h and 0 <= default_c <= w - crop_w:
             return default_r, default_c
         else:
            raise ValueError("Không tìm thấy tọa độ crop phù hợp sau khi duyệt.")


    return best_crop_coords

# --- Hàm xử lý chính cho một ảnh (không in, chỉ lưu file) ---
def process_single_image(
    image_path: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    center_lat: float,
    center_lon: float,
    output_path: str,
    img_height: int = 400,
    img_width: int = 400,
    crop_size: int = 303,
    min_angle: float = 0.0,
    max_angle: float = 0.0 # Sử dụng max_angle từ ví dụ trước
) -> bool:
    if not os.path.exists(image_path):
        # Không in lỗi, chỉ trả về False để hàm gọi quyết định
        return False
    # Đọc ảnh giữ nguyên kênh alpha nếu có
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if original_image is None:
        return False # Lỗi đọc ảnh
    actual_h, actual_w = original_image.shape[:2]
    # Không cần kiểm tra kích thước nếu bạn chắc chắn chúng đúng
    # Chuyển đổi và tính toán tọa độ
    center_row_orig, center_col_orig = latlon_to_pixel(
        center_lat, center_lon, min_lat, max_lat, min_lon, max_lon, actual_h, actual_w
    )
    # Quay ảnh
    random_angle = random.uniform(min_angle, max_angle)
    rotation_center_coords = (actual_w / 2, actual_h / 2)
    rotated_image, rotation_matrix = rotate_image(original_image, random_angle, center=rotation_center_coords)
    # Tính tọa độ tâm mới và đảm bảo nằm trong ảnh
    center_col_new, center_row_new = transform_point((center_col_orig, center_row_orig), rotation_matrix)
    center_row_new = max(0, min(center_row_new, actual_h - 1))
    center_col_new = max(0, min(center_col_new, actual_w - 1))
    # Tìm tọa độ crop tốt nhất
    best_r, best_c = find_max_sum_crop(rotated_image, center_row_new, center_col_new, crop_size)
    # Cắt ảnh
    final_crop = rotated_image[best_r : best_r + crop_size, best_c : best_c + crop_size]
    # Kiểm tra kích thước ảnh cuối cùng trước khi lưu
    if final_crop.shape[0] != crop_size or final_crop.shape[1] != crop_size:
         # Lỗi cắt không đúng kích thước
         return False
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(output_path)
    if output_dir: # Chỉ tạo nếu có đường dẫn thư mục (không phải file ở thư mục hiện tại)
        os.makedirs(output_dir, exist_ok=True)
        
    # if os.path.exists(output_path):
    #     os.remove(output_path)  # Xóa ảnh cũ
    # Lưu ảnh kết quả
    save_success = cv2.imwrite(output_path, final_crop)
    return save_success

# --- Hàm đọc CSV và điều phối xử lý ---
def process_csv_and_generate_images(
    csv_path: str,
    base_image_dir: str,
    output_dir: str,
    crop_size: int = 303,
    img_height: int = 400,
    img_width: int = 400,
    min_angle: float = 0.0,
    max_angle: float = 15.0
):
    df = pd.read_csv(csv_path, 
                       parse_dates=['most recent time'],
                       usecols=['International number ID', 'Name of the storm', 'Time of analysis',
                                'Grade', 'Latitude of the center', 'Longitude of the center',
                                'Central pressure', 'Maximum sustained wind speed',
                                'The longest radius of 50kt winds or greater', 
                                'Bands and number', 'min_lon', 'max_lon', 'min_lat', 'max_lat', 'most recent time'])

    processed_count = 0
    error_count = 0

    # Lặp qua từng dòng trong DataFrame
    for index, row in df.iterrows():
        try:
            # Trích xuất thông tin
            storm_id = row['International number ID']
            center_lat = float(row['Latitude of the center'])
            center_lon = float(row['Longitude of the center'])
            min_lon = float(row['min_lon'])
            max_lon = float(row['max_lon'])
            min_lat = float(row['min_lat'])
            max_lat = float(row['max_lat'])
            
            most_recent_time = row["most recent time"]  # Chuyển thành datetime object# Tách thông tin từ datetime
            YYYY = most_recent_time.strftime("%Y")
            MM = most_recent_time.strftime("%m")
            DD = most_recent_time.strftime("%d")
            hh = most_recent_time.strftime("%H")
            mm = most_recent_time.strftime("%M")

            most_recent_time -= timedelta(minutes=20)
            mm1 = most_recent_time.minute
            base_time_0 = f"{YYYY}{MM}{DD}{hh}{mm}"
            base_time_1 = f"{YYYY}{MM}{DD}{hh}{mm}"

            # Xây dựng đường dẫn file ảnh input
            image_filename = f"conbaomini_{base_time_0}.png"
            input_image_path = f'{base_image_dir}{storm_id}/image/{base_time_0}/{image_filename}'
            # Xây dựng đường dẫn file ảnh output
            output_filename = f"conbaofinal_{base_time_1}.png"
            output_image_path = f'{OUTPUT_DIRECTORY}{storm_id}/image/{base_time_1}/{output_filename}'
            # output_image_path = 'out.png'

            # Gọi hàm xử lý cho ảnh này
            success = process_single_image(
                image_path=input_image_path,
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                output_path=output_image_path,
                img_height=img_height,
                img_width=img_width,
                crop_size=crop_size,
                min_angle=min_angle,
                max_angle=max_angle
            )

            if success:
                print(output_image_path)
                processed_count += 1
            else:
                error_count += 1
                # Không in lỗi ở đây để giữ im lặng

        except (ValueError, KeyError, TypeError) as e:
            error_count += 1
            continue # Bỏ qua dòng bị lỗi

    # In thông báo tổng kết cuối cùng
    print(f"Hoàn tất xử lý.")
    print(f" - Đã xử lý thành công: {processed_count} ảnh.")
    print(f" - Gặp lỗi hoặc bỏ qua: {error_count} dòng/ảnh.")
    print(f" - Ảnh kết quả đã được lưu vào thư mục: {output_dir}")

# --- Phần thực thi chính ---
if __name__ == "__main__":
    # ----- CẤU HÌNH -----
    # CSV_FILE_PATH = 'val_add.csv'  # Đường dẫn đến file CSV của bạn
    CSV_FILE_PATH = 'test.csv'  # Đường dẫn đến file CSV của bạn
    BASE_IMAGE_DIRECTORY = '/datausers3/kttv/tien/ClassificationProject/mtsat_data/working/' # Thư mục gốc chứa ảnh
    OUTPUT_DIRECTORY = '/datausers3/kttv/tien/ClassificationProject/my_mtsat_data/working/'        # Thư mục lưu ảnh kết quả
    CROP_SIZE = 303                               # Kích thước ảnh cắt
    MIN_ROTATION_ANGLE = 0.0                      # Góc quay tối thiểu
    MAX_ROTATION_ANGLE = 0.0                    # Góc quay tối đa
    # ----- KẾT THÚC CẤU HÌNH -----

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print(f"Bắt đầu xử lý file CSV: {CSV_FILE_PATH}")
    print(f"Thư mục ảnh gốc: {BASE_IMAGE_DIRECTORY}")
    print(f"Thư mục lưu kết quả: {OUTPUT_DIRECTORY}")

    # Gọi hàm xử lý chính
    process_csv_and_generate_images(
        csv_path=CSV_FILE_PATH,
        base_image_dir=BASE_IMAGE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        crop_size=CROP_SIZE,
        min_angle=MIN_ROTATION_ANGLE,
        max_angle=MAX_ROTATION_ANGLE
    )