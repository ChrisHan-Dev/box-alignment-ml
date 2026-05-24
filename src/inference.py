import cv2
import math
from ultralytics import YOLO

# 1. Khởi tạo mô hình YOLO OBB (Sẽ tự động chạy trên GPU RTX 4070 Super)
model = YOLO("yolov8n-obb.pt")

def analyze_box_alignment(image_path, max_allow_deviation=5.0):
    """
    Hàm quét ảnh Front-View từ camera, phát hiện hộp 
    và tính toán góc nghiêng trực diện của hộp so với mặt băng chuyền.
    """
    print(f"[INFO] Chạy AI phân tích ảnh: {image_path}")
    
    # Thực hiện dự đoán trên GPU (device=0)
    results = model(image_path, device=0)
    flag_teleoperator = False
    
    # Kiểm tra nếu phát hiện được khung hình chữ nhật xoay (OBB)
    if results[0].obb is not None and len(results[0].obb.data) > 0:
        for box in results[0].obb.data:
            # Định dạng OBB: [x_ctr, y_ctr, w, h, angle_rad, conf, class_id]
            _, _, _, _, angle_rad, conf, _ = box.tolist()
            
            # 2. Chuyển đổi góc từ Radian sang Độ (Degree)
            angle_deg = math.degrees(angle_rad)
            
            # 3. Tính toán độ lệch nghiêng (Front-view deviation)
            # YOLO OBB định nghĩa góc từ 0 đến 180 độ hoặc tương đương.
            # Ta đưa về độ lệch so với phương thẳng đứng/nằm ngang chuẩn.
            deviation = min(angle_deg % 90, 90 - (angle_deg % 90))
            
            # 4. Đánh giá trạng thái căn chỉnh
            if deviation > max_allow_deviation:
                status = f"BỊ NGHIÊNG/LỆCH CHÉO ({deviation:.2f}°)"
                flag_teleoperator = True
            else:
                status = f"THẲNG HÀNG CHUẨN ({deviation:.2f}°)"
                
            print(f" -> Phát hiện vật thể (Độ tự tin: {conf*100:.1f}%) | Trạng thái: {status}")
            
    else:
        print("[WARN] Không tìm thấy hộp hoặc vật thể nào trong tầm nhìn của mô hình mặc định.")
        
    # 5. Kích hoạt định tuyến đến Teleoperator nếu phát hiện lỗi đặt hộp
    if flag_teleoperator:
        trigger_teleoperation_route()

def trigger_teleoperation_route():
    print("\n[ALERT] !!! CẢNH BÁO: HỘP TRÊN BĂNG CHUYỀN BỊ NGHIÊNG/LỆCH QUÁ MỨC !!!")
    print("[ALERT] Đang gửi tín hiệu yêu cầu Teleoperator thủ công can thiệp...\n")

if __name__ == "__main__":
    # Kích hoạt hàm kiểm tra với tấm ảnh bạn đã để trong thư mục src/images/
    analyze_box_alignment("src/images/test_box.webp")