# Các điểm cần nhóm thống nhất – phần Data

> Gom từ kết quả EDA trong [DATA.md](DATA.md). Cột **Xem ở DATA.md** chỉ mục chứa số liệu và lập luận; "mục 3, #n" là dòng số n trong bảng quyết định xử lý ở mục 3.
> Cột **Đề xuất** là ý kiến ban đầu, nhóm có thể chọn khác. Điền **Kết luận của nhóm** và **Người quyết / Ngày** khi đã chốt.
> Mục A chặn việc viết `clean.py` và `split.py`, nên chốt trước.

---

## A. Cần chốt trước khi viết code pipeline

| # | Điểm cần thống nhất | Xem ở DATA.md | Phương án | Đề xuất | Kết luận của nhóm | Người quyết / Ngày |
|---|---|---|---|---|---|---|
| 1 | **Dòng trùng** (410.660 dòng, 2,5%) | Mục 2.5 (số liệu); mục 3, #9 (hai phương án) | (a) Loại trước khi chia tập; (b) giữ, gom các bản trùng vào cùng một tập | (a). Cái giá: FTP-BruteForce mất 80% mẫu, SlowHTTPTest mất 86% | | |
| 2 | **Ngày nào là train, ngày nào là mô phỏng vận hành** | Mục 5 (bảng đề xuất); mục 2.3 (tấn công theo ngày) | Train: 14/02, 15/02, 16/02, 20/02, 21/02. `production_sim`: 22/02, 23/02, 28/02, 01/03, 02/03 (Web, Infiltration, Bot là loại tấn công mới) | Như cột phương án, vì tạo được kịch bản Drift tự nhiên | | |
| 3 | **Cách chia train/val/test trong giai đoạn đầu** | Mục 5 (đoạn "Lý do"); mục 2.8 (vì sao giờ và thứ tự dòng không tin được) | Theo khối liên tiếp theo thứ tự dòng; theo giờ suy luận; hoặc ngẫu nhiên phân tầng sau khi loại trùng | Chưa chốt được: thứ tự dòng và giờ trong dữ liệu không đáng tin | | |
| 4 | **Giả định AM/PM** (giờ 1 đến 7 là buổi chiều, cộng 12) | Mục 2.8; mục 9 (rủi ro "mất AM/PM") | Chấp nhận giả định để chia theo giờ; hoặc bỏ hẳn việc chia theo giờ | Chỉ dùng cho biểu đồ, không dựa vào để chia tập | | |
| 5 | **Xử lý `NaN`/`inf`** (luồng thời lượng 0 nên chia cho 0) | Mục 2.4 (số liệu); mục 3, #3 | (a) Loại dòng (dưới 1,2%); (b) đổi `inf` thành `NaN` rồi điền 0 | (a) | | |
| 6 | **Giá trị −1 ở `Init Fwd/Bwd Win Byts`** (27% và 51% số dòng) | Mục 2.4; mục 3, #5 | Giữ nguyên; hoặc thêm cột cờ "không có cửa sổ" | Giữ, thêm cờ nếu cần | | |
| 7 | **Lấy mẫu Benign ngày 20/02** (7,37 triệu dòng, 49% dữ liệu) | Mục 2.1, 2.3; mục 3, #13; mục 5 (phần "Việc khác") | Giữ hết; hoặc lấy mẫu theo một tỉ lệ | Nên lấy mẫu; tỉ lệ cụ thể nhóm quyết | | |

## B. Nhãn và đặc trưng

| # | Điểm cần thống nhất | Xem ở DATA.md | Phương án | Đề xuất | Kết luận của nhóm | Người quyết / Ngày |
|---|---|---|---|---|---|---|
| 8 | **Bài toán** | Mục 2.2 (phân bố nhãn và mất cân bằng) | Nhị phân trước, đa lớp là phần mở rộng (đúng README) | Giữ như README | | |
| 9 | **Lớp Web** (611, 230, 87 mẫu) khi làm đa lớp | Mục 2.2; mục 3, #11 | Gộp thành một lớp "Web Attack"; hoặc loại khỏi đa lớp | Gộp | | |
| 10 | **Chuẩn hóa tên nhãn** (`DDOS`/`DDoS`, `Infilteration`) | Mục 2.2; mục 3, #10 | Sửa chính tả; hoặc giữ tên gốc | Chuẩn hóa, lưu `label_map.json` | | |
| 11 | **Đặc trưng thừa** (34 cặp tương quan ≥ 0,99; 8 cột hằng số) | Mục 2.4 (cột hằng số); mục 2.6 (tương quan); mục 3, #6 và #8 | Ngưỡng 0,99 hay thấp hơn (ví dụ 0,95); chọn cột đại diện nào | Bỏ 8 cột hằng số trước; ngưỡng tương quan nhóm bàn | | |
| 12 | **`Dst Port` và `Protocol`** có dùng làm đặc trưng không | Mục 3, #14 (chưa có số liệu EDA cho hai cột này) | Giữ; hoặc bỏ (cổng đích có thể khiến mô hình học theo dịch vụ thay vì hành vi) | Cần thử huấn luyện có và không có để so sánh | | |
| 13 | **Cân bằng lớp** | Mục 2.2 (kết luận về mất cân bằng); mục 3, #12 | `class_weight`/`scale_pos_weight`; undersample; hoặc SMOTE (chỉ trên train) | `class_weight` trước | | |
| 14 | **Scaler** | Chưa có số liệu riêng trong DATA.md; mục 4 (bước `preprocess`) | `RobustScaler`; hoặc `StandardScaler` | `RobustScaler`, vì phân phối có đuôi dài (xem biểu đồ mục 2.7) | | |

## C. Phối hợp giữa các thành viên

| # | Điểm cần thống nhất | Liên quan | Xem ở DATA.md | Kết luận của nhóm | Người quyết / Ngày |
|---|---|---|---|---|---|
| 15 | **Chỉ số ưu tiên và ngưỡng** (Recall, FPR; ngưỡng Drift; ngưỡng tái huấn luyện) | Huấn luyện, giám sát | Mục 2.2 (chỉ nêu "không dùng Accuracy"). Ngưỡng cụ thể **không có trong DATA.md**, xem README mục 6 và 8 | | |
| 16 | **Tập tham chiếu cho Evidently** (đề xuất: tập train) và cách phát lại `production_sim` (kích thước mỗi lô, thứ tự) | Giám sát | Mục 4 (sơ đồ); mục 5 (`production_sim`) | | |
| 17 | **Ai giữ và cập nhật `features.json`, `scaler.pkl`** (API phải dùng đúng bộ này) | API | Mục 9 (rủi ro "train–serve skew"); mục 4 (bước `preprocess`) | | |
| 18 | **Có dùng DVC không**, lưu dữ liệu ở đâu, MLflow ghi những gì về dữ liệu | Toàn nhóm | Mục 6 | | |
| 19 | **Định dạng và đường dẫn dữ liệu đã xử lý** (đề xuất: Parquet trong `data/processed/`) | Huấn luyện, Airflow | Mục 4 (sơ đồ); mục 7 (cấu trúc thư mục) | | |
| 20 | **Có commit biểu đồ và báo cáo lên Git không** (`reports/`, `docs/`); dữ liệu thô thì không commit | Quản lý mã nguồn | Mục 6 (bảng "Dữ liệu raw"); mục 7 | | |

## D. Chấp nhận như một hạn chế (nhóm đồng ý ghi nhận)

| # | Hạn chế | Xem ở DATA.md | Hành động | Kết luận của nhóm |
|---|---|---|---|---|
| 21 | 7 file có đúng 1.048.575 dòng, nghi bị cắt ở giới hạn của Excel (chưa kiểm chứng) | Mục 2.1; mục 9; mục 11 (Data Card, "Giới hạn đã biết") | Chấp nhận, hoặc tìm bản không bị cắt | |
| 22 | Giấy phép và điều khoản sử dụng dữ liệu của CIC chưa được kiểm tra | Mục 9; mục 11 (Data Card, "Giấy phép") | Kiểm tra trước khi chia sẻ lại dữ liệu | |
