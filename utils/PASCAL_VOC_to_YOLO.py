import os
import xml.etree.ElementTree as ET
from glob import glob

def convert_pascal_voc_to_yolo(xml_file, class_list):
    # Đọc file XML
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Lấy kích thước ảnh
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    
    # List để lưu các annotation theo định dạng YOLO
    yolo_annotations = []
    
    # Xử lý từng object trong file XML
    for obj in root.findall('object'):
        # Lấy tên class
        class_name = obj.find('name').text
        
        # Nếu class chưa có trong list thì thêm vào
        if class_name not in class_list:
            class_list.append(class_name)
            
        # Lấy class_id
        class_id = class_list.index(class_name)
        
        # Lấy tọa độ bbox
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Chuyển đổi sang định dạng YOLO
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        
        # Thêm vào list annotations
        yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_annotations.append(yolo_annotation)
    
    return yolo_annotations

def main():
    # Đường dẫn thư mục chứa file XML
    xml_dir = "datasets/Sand/Sand/Sand_PASCAL_VOC"  # Thay đổi đường dẫn này
    
    # Đường dẫn thư mục output cho file txt
    output_dir = "Sand/Sand/Sand_YOLO_darknet2"  # Thay đổi đường dẫn này
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # List để lưu tất cả các class
    class_list = []
    
    # Lấy tất cả file XML trong thư mục
    xml_files = glob(os.path.join(xml_dir, "*.xml"))
    
    # Xử lý từng file XML
    for xml_file in xml_files:
        # Tạo tên file output
        filename = os.path.splitext(os.path.basename(xml_file))[0]
        txt_file = os.path.join(output_dir, filename + ".txt")
        
        # Chuyển đổi và lưu annotations
        yolo_annotations = convert_pascal_voc_to_yolo(xml_file, class_list)
        
        # Ghi vào file txt
        with open(txt_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    # Ghi danh sách class vào file classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_list))
    
    print(f"Đã chuyển đổi {len(xml_files)} file")
    print(f"Tổng số class: {len(class_list)}")
    print("Danh sách class:", class_list)

if __name__ == "__main__":
    main()