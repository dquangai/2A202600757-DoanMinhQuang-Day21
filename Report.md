# Báo Cáo Thực Hành MLOps - Day 21

**Họ và tên:** Đoàn Minh Quang  
**Mã học viên (MHV):** 2A202600757  
**Khóa học:** AI In Action - VinUni  

---

## 1. Tóm Tắt Các Công Việc Đã Thực Hiện

Dưới đây là chi tiết các hạng mục đã hoàn thành trong bài thực hành này, bao gồm cấu hình môi trường, theo dõi thí nghiệm, quản lý dữ liệu, kiểm thử và thiết lập pipeline CI/CD tự động hóa.

### Bước 1: Thực Nghiệm Cục Bộ và Theo Dõi Thí Nghiệm
- **Thiết lập Môi trường & Thư viện:** Đã khởi tạo môi trường ảo `.venv` và cài đặt các thư viện trong `requirements.txt`. Downgrade `setuptools==69.5.1` để tương thích với `mlflow` trên Python 3.12 (khắc phục lỗi import `pkg_resources`).
- **Phân tách Dữ liệu:** Đã chạy `generate_data.py` phân tách thành các tập: `train_phase1.csv` (2998 mẫu), `eval.csv` (500 mẫu) và `train_phase2.csv` (2998 mẫu).
- **Thực nghiệm 3 lần chạy với MLflow cục bộ:**
  - **Lần chạy 1 (Random Forest):** 
    - Tham số: `n_estimators=100`, `max_depth=5`, `min_samples_split=2`
    - Kết quả: **Accuracy = 0.5640**, **F1-Score = 0.5534**
  - **Lần chạy 2 (Gradient Boosting):**
    - Tham số: `model_type=gradient_boosting`, `n_estimators=150`, `max_depth=4`, `min_samples_split=3`
    - Kết quả: **Accuracy = 0.6120**, **F1-Score = 0.6102**
  - **Lần chạy 3 (Random Forest tối ưu hơn):**
    - Tham số: `model_type=random_forest`, `n_estimators=200`, `max_depth=12`, `min_samples_split=5`
    - Kết quả: **Accuracy = 0.6560**, **F1-Score = 0.6547**
- **Dữ liệu huấn luyện tích hợp:** Khi gộp cả Phase 1 + Phase 2 (giai đoạn huấn luyện liên tục), độ chính xác của Random Forest tăng đáng kể lên **0.7540** (đạt và vượt ngưỡng yêu cầu 0.70).

### Bước 2: Quản Lý Phiên Bản Dữ Liệu Bằng DVC
- **Khởi tạo DVC:** Đã chạy `dvc init` để thiết lập DVC cho dự án.
- **Cấu hình Remote Storage:** Cấu hình local remote (`myremote` trỏ đến thư mục `../dvc_storage_local` ngoài git tree) để hỗ trợ quá trình kiểm thử cục bộ chạy `dvc push` và `dvc pull` trơn tru. Trên CI/CD, remote sẽ được gán động sang Google Cloud Storage (GCS) bằng secret credential.
- **Theo dõi dữ liệu:** Đã thêm các tập tin CSV vào DVC:
  - `data/train_phase1.csv.dvc`
  - `data/eval.csv.dvc`
  - `data/train_phase2.csv.dvc`
- **Lưu trữ dữ liệu:** Đẩy dữ liệu thành công lên kho chứa cục bộ thông qua `dvc push` và commit các file `.dvc` cùng `.gitignore` vào git repository.

### Bước 3: Phát Triển REST API & Kiểm Thử
- **REST API (`src/serve.py`):** Viết API suy luận bằng FastAPI hỗ trợ:
  - Endpoint `/health` kiểm tra sức khỏe của server, trả về `{"status": "ok"}`.
  - Endpoint `/predict` nhận đầu vào JSON gồm 12 đặc trưng, phân loại rượu vang theo nhãn chất lượng `thap`, `trung_binh`, `cao`.
  - Tích hợp tải tự động mô hình `model.pkl` từ GCS về thư mục cục bộ của VM khi API khởi chạy.
- **Unit Tests (`tests/test_train.py`):** Viết 3 bộ kiểm thử tự động sử dụng `pytest` chạy trên dữ liệu giả lập (mock data):
  - `test_train_returns_float`: Đảm bảo hàm `train` trả về một số thực trong khoảng `[0.0, 1.0]`.
  - `test_metrics_file_created`: Đảm bảo file `outputs/metrics.json` được sinh ra và chứa các trường `accuracy`, `f1_score`.
  - `test_model_file_created`: Đảm bảo file model `models/model.pkl` được lưu trữ thành công.
  - **Kết quả chạy thử nghiệm cục bộ:** Bộ 3 tests đều **PASSED** (23.70s).

### Bước 4: Tự Động Hóa Pipeline CI/CD (`.github/workflows/mlops.yml`)
Thiết lập quy trình CI/CD hoàn chỉnh trên GitHub Actions kích hoạt mỗi khi có thay đổi ở code hoặc dữ liệu (`.dvc`), bao gồm 4 Jobs:
1. **Unit Test:** Cài đặt dependencies và chạy bộ kiểm thử `pytest tests/ -v`.
2. **Train:** Xác thực tài khoản Cloud, kéo dữ liệu gốc bằng `dvc pull`, thực hiện huấn luyện mô hình, ghi log siêu tham số/chỉ số lên MLflow, lưu `metrics.json` và upload mô hình `model.pkl` lên GCS.
3. **Eval (Quality Gate):** Đọc độ chính xác mới, ngăn chặn triển khai nếu accuracy dưới 0.70 và so sánh độ chính xác hiện tại với lịch sử để ngăn chặn suy thoái mô hình (degradation check).
4. **Deploy:** SSH kết nối trực tiếp vào GCE VM, chạy khởi động lại service `mlops-serve` và kiểm tra sức khỏe API thông qua curl.

---

## 2. Các Tính Năng Bonus Đã Hoàn Thành (Full Điểm & Điểm Cộng)

Để đạt điểm số tối đa cho bài lab, các tính năng bonus nâng cao đã được tích hợp trực tiếp vào dự án:

### [Bonus 1] Theo Dõi MLflow Từ Xa (Remote Tracking với DagsHub)
Tích hợp tự động trong `.github/workflows/mlops.yml`. Nếu cấu hình các secrets `MLFLOW_TRACKING_USERNAME` và `MLFLOW_TRACKING_PASSWORD`, pipeline sẽ tự động chuyển hướng tracking URI từ local sang DagsHub server:
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/${{ github.repository }}.mlflow
```
Tất cả metrics, parameters và model artifact sẽ được lưu trữ tập trung trên DagsHub để dễ dàng so sánh trực quan.

### [Bonus 2] Hỗ Trợ Nhiều Thuật Toán
Hàm huấn luyện `train()` trong [src/train.py](file:///d:/Users/quand/2A202600757-DoanMinhQuang-Day21/src/train.py) đã được nâng cấp để hỗ trợ nhiều mô hình dựa trên tham số cấu hình `model_type` trong `params.yaml`:
- **Random Forest:** `model_type: random_forest`
- **Gradient Boosting:** `model_type: gradient_boosting`
- **Logistic Regression:** `model_type: logistic_regression` (với tùy chọn tăng số lượng vòng lặp tối đa `max_iter` để hội tụ).

### [Bonus 3] Báo Cáo Hiệu Suất Tự Động
Trong [src/train.py](file:///d:/Users/quand/2A202600757-DoanMinhQuang-Day21/src/train.py), chương trình tự động sinh Confusion Matrix (Ma trận nhầm lẫn) và Precision/Recall/F1-score cho từng lớp chất lượng rượu (0, 1, 2). Kết quả được ghi nhận vào file [outputs/report.txt](file:///d:/Users/quand/2A202600757-DoanMinhQuang-Day21/outputs/report.txt). File này cũng được đóng gói thành artifact của GitHub Actions workflow để tải về trực tiếp từ trang Run.

### [Bonus 4] Hoàn Trả Về Phiên Bản Trước (Degradation Safety Gate)
Tích hợp cơ chế kiểm soát chất lượng mô hình chặt chẽ tại Job **Eval** của pipeline CI/CD:
- Đọc file metrics lịch sử của bản deploy trước đó lưu trên GCS.
- So sánh `new_accuracy` với `old_accuracy`.
- **Nếu `new_accuracy < old_accuracy`:** Hủy bỏ quá trình deploy lập tức (`raise SystemExit`) để bảo vệ môi trường Production khỏi việc suy giảm hiệu suất.
- **Nếu chất lượng cải thiện:** Cập nhật độ chính xác mới làm mốc tham chiếu cho các lần chạy sau trên GCS.

### [Bonus 5] Cảnh Báo Lệch Lạc Dữ Liệu (Class Skew Warning)
Trong [src/train.py](file:///d:/Users/quand/2A202600757-DoanMinhQuang-Day21/src/train.py), chương trình tự động thống kê phân phối của cột mục tiêu `target` trên tập huấn luyện (tỷ lệ của từng lớp 0, 1, 2). 
- In cảnh báo `WARNING: Lop X chiem Y% (< 10%) tong so mau!` ra màn hình nếu có bất kỳ lớp nào chiếm tỷ lệ dưới 10% mẫu (dấu hiệu lệch dữ liệu nghiêm trọng).
- Ghi nhận chi tiết tỷ lệ phần trăm phân phối này vào `outputs/metrics.json` để phục vụ giám sát dữ liệu (data drift & skew monitoring).

---

Báo cáo được chuẩn bị đầy đủ và sẵn sàng cho việc nộp bài đánh giá!
