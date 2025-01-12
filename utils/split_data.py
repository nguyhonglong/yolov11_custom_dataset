import os
import shutil
import random
from glob import glob

def create_directories(base_path):
    # Tạo các thư mục cần thiết
    dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels',
        'test/images', 'test/labels'
    ]
    for dir_path in dirs:
        os.makedirs(os.path.join(base_path, dir_path), exist_ok=True)

def split_data(source_path, destination_path, train_ratio=0.7, val_ratio=0.2):
    # Lấy danh sách các file ảnh
    image_files = glob(os.path.join(source_path, "*.jpg"))
    
    # Xáo trộn ngẫu nhiên danh sách file
    random.shuffle(image_files)
    
    # Tính số lượng file cho mỗi tập
    num_files = len(image_files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    
    # Chia thành các tập
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]
    
    # Hàm copy file
    def copy_files(files, subset):
        for img_path in files:
            # Lấy tên file không có phần mở rộng
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Đường dẫn file label tương ứng
            label_path = os.path.join(source_path, "Fog_YOLO_darknet2", f"{name_without_ext}.txt")
            
            # Copy ảnh
            dst_img = os.path.join(destination_path, f"{subset}/images", filename)
            shutil.copy2(img_path, dst_img)
            
            # Copy label nếu tồn tại
            if os.path.exists(label_path):
                dst_label = os.path.join(destination_path, f"{subset}/labels", f"{name_without_ext}.txt")
                shutil.copy2(label_path, dst_label)
    
    # Copy files vào các thư mục tương ứng
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    # In thống kê
    print(f"Tổng số ảnh: {num_files}")
    print(f"Số ảnh train: {len(train_files)}")
    print(f"Số ảnh validation: {len(val_files)}")
    print(f"Số ảnh test: {len(test_files)}")

def main():
    # Đường dẫn thư mục nguồn chứa ảnh và labels
    source_path = "path/to/your/source/dataset"  # Replace with your source dataset path
    
    # Đường dẫn thư mục đích để lưu các tập train/val/test
    destination_path = "D:/yolov11_custom_dataset/splitted_dataset"
    
    # Tạo các thư mục cần thiết
    create_directories(destination_path)
    
    # Chia dữ liệu
    split_data(source_path, destination_path)

if __name__ == "__main__":
    main()
