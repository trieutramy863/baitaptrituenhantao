# baitaptrituenhantao
# link youtube
https://youtu.be/2Bm3dbhklok
# Hướng dẫn cài đặt và chạy chương trình
🔹 Bước 1: Cài đặt Anaconda

Truy cập: https://www.anaconda.com

Tải và cài đặt Anaconda

Sau khi cài xong, mở Anaconda Prompt

🔹 Bước 2: Tạo môi trường Python 3.10

Nhập lệnh:

conda create -n ai_env python=3.10

→ Nhấn y để xác nhận cài đặt

🔹 Bước 3: Kích hoạt môi trường
conda activate ai_env

→ Nếu thấy (ai_env) phía trước là thành công ✅

🔹 Bước 4: Cài đặt các thư viện cần thiết

Nhập lần lượt:

<pre>'''pip install tensorflow
pip install flask
pip install pillow
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install opencv-python </pre>
🔹 Bước 5: Mở thư mục chứa project

Ví dụ project nằm ở ổ D:
cd D:\PhanLoaiRacThai

🔹 Bước 6: Mở project bằng PyCharm hoặc VS Code

Nếu dùng PyCharm:

Chọn New Project

Chọn Conda Environment

Chọn môi trường ai_env

🔹 Bước 7: Chuẩn bị dữ liệu (dataset)

Đảm bảo thư mục dataset có dạng:
<pre>
dataset/
 ├── huuco/
 ├── nguyhai/
 └── taiche/
</pre>
👉 Mỗi thư mục chứa ảnh tương ứng:

huuco → rác hữu cơ

nguyhai → rác nguy hại

taiche → rác tái chế

🔹 Bước 8: Huấn luyện mô hình (nếu chưa có model)

Chạy lệnh:

python train_model.py

→ Sau khi chạy xong sẽ tạo file:
model.h5

🔹 Bước 9: Tạo thư mục lưu ảnh upload
mkdir static/upload

🔹 Bước 10: Chạy chương trình
python app.py

→ Nếu thành công sẽ hiển thị:

Running on http://127.0.0.1:5000/

🔹 Bước 11: Mở trình duyệt và sử dụng

Truy cập:http://127.0.0.1:5000/

→ Thực hiện:

Upload ảnh rác

Hệ thống sẽ:

Phân loại (Hữu cơ / Nguy hại / Tái chế)

Hiển thị độ chính xác (%)
