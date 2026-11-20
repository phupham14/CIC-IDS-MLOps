# Serving & Infra – Triển khai API, Docker, Prometheus và Grafana (TV3)

> Phạm vi: đóng gói hạ tầng bằng Docker Compose (`mlflow`, `api`, `pushgateway`, `prometheus`, `grafana`),
> xây dựng REST API phục vụ mô hình (`src/api/`), ghi log dự đoán ra Parquet theo đợt (`logs/predictions/`),
> thiết lập Dashboard & Cảnh báo trên Grafana (`deploy/grafana/`), và script phát lại dữ liệu vận hành (`src/api/replay.py`).
> Hợp đồng giao tiếp với TV4: xem [INTERFACE.md](INTERFACE.md).

## Mục lục

1. [Cách chạy nhanh](#1-cách-chạy-nhanh)
2. [Kiến trúc các dịch vụ Docker Compose](#2-kiến-trúc-các-dịch-vụ-docker-compose)
3. [Thiết kế FastAPI và các Endpoint](#3-thiết-kế-fastapi-và-các-endpoint)
4. [Ghi log dự đoán và nhãn trễ (Parquet)](#4-ghi-log-dự-đoán-và-nhãn-trễ-parquet)
5. [Chỉ số Prometheus và Dashboard Grafana](#5-chỉ-số-prometheus-và-dashboard-grafana)

---

## 1. Cách chạy nhanh

### 1.1. Chạy kiểm thử tự động (Unit & Integration Tests)
Không cần bật Docker và không cần tải dữ liệu gốc (test tự sinh dữ liệu tổng hợp cùng schema 70 đặc trưng):

```bash
python -m pytest tests/api -v
```

### 1.2. Khởi động toàn bộ hạ tầng bằng Docker Compose

```bash
# 1. Build image và khởi động 5 container chạy ngầm
docker compose up -d --build

# 2. Kiểm tra trạng thái các container
docker compose ps
```

| Dịch vụ | Container | Cổng | Địa chỉ truy cập | Ghi chú |
|---|---|---|---|---|
| **MLflow Server** | `cic-ids-mlflow` | `5000` | http://localhost:5000 | SQLite backend + `--serve-artifacts` trên volume `./mlruns` |
| **FastAPI Serving** | `cic-ids-api` | `8000` | http://localhost:8000/docs | Swagger UI, tự nạp `cic-ids-binary@champion` |
| **Pushgateway** | `cic-ids-pushgateway` | `9091` | http://localhost:9091 | Nhận chỉ số drift/recall theo lô từ `drift.py` (TV4) |
| **Prometheus** | `cic-ids-prometheus` | `9090` | http://localhost:9090 | Scrape `api:8000/metrics` và `pushgateway:9091` mỗi 10s |
| **Grafana** | `cic-ids-grafana` | `3000` | http://localhost:3000 | Tài khoản: `admin` / `admin` (đã nạp sẵn Dashboard & Alerts) |

### 1.3. Đăng ký mô hình `@champion` lên MLflow Server

Trỏ biến môi trường `MLFLOW_TRACKING_URI` vào MLflow Server trước khi huấn luyện hoặc đăng ký mô hình:

```bash
# PowerShell:
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
# Bash / Linux:
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Cách A: Huấn luyện từ dữ liệu thật trong data/processed/
python -m src.models.train --models xgb --tune --register
python -m src.models.compare --promote

# Cách B: Tạo nhanh 1 model mẫu từ dữ liệu tổng hợp (khi máy chưa có data/processed/)
python -c "import yaml, mlflow; from tests.models.test_models import FEATURES, make_frame; from src.models.data import Split; from src.models import estimators as E, train as T, registry as R; cfg=yaml.safe_load(open('models.yaml',encoding='utf-8')); R.setup(cfg); df=make_frame(300,42); sp=Split(df[FEATURES],df['y_binary'].to_numpy(),df['Label'].to_numpy(),df['meta_day'].to_numpy()); mlflow.start_run(run_name='demo-xgb'); est=E.fit('xgb',E.build('xgb',{'n_estimators':15},1.0),sp,sp); info=T.log_model('xgb',est,FEATURES,0.5,'data/preprocessing',{'feature_set':'all'}); mv=R.register(info.model_uri,cfg['registry']['model_name'],'challenger'); R.promote(cfg['registry']['model_name'],mv.version,cfg['registry']); mlflow.end_run(); print('Champion version:', mv.version)"
```

### 1.4. Kiểm tra API và phát lại dữ liệu (`replay.py`)

```bash
# Yêu cầu API nạp ngay bản @champion mới nhất (hoặc đợi tự động mỗi 60s)
curl.exe -X POST http://localhost:8000/reload

# Xem thông tin phiên bản model đang phục vụ
curl.exe http://localhost:8000/model

# Phát lại dữ liệu production_sim gửi theo lô vào API và ghi nhãn thật trễ 1 lô
python -m src.api.replay --scenario s1_new_attacks --api-url http://localhost:8000 --max-batches 5 --sleep 1.0
```

Khi kết thúc thử nghiệm, tắt cụm Docker bằng lệnh:
```bash
docker compose down
```

---

## 2. Kiến trúc các dịch vụ Docker Compose

File [docker-compose.yml](../docker-compose.yml) và thư mục `deploy/` được thiết kế để toàn nhóm dùng chung không cần cài đặt thủ công:

- **Dùng chung Image giữa `mlflow` và `api` (`cic-ids-app:latest`):** Build từ [deploy/docker/Dockerfile.api](../deploy/docker/Dockerfile.api) dựa trên `python:3.11-slim` + `libgomp1` + `requirements.txt`. Nhờ đó phiên bản MLflow giữa server, API và môi trường train luôn đồng nhất 100%.
- **MLflow Artifact Proxy (`--serve-artifacts`):** Mọi container trong Docker (`MLFLOW_TRACKING_URI=http://mlflow:5000`) và máy host (`http://localhost:5000`) đều tải/lưu model qua giao thức HTTP của MLflow mà không bị lệch đường dẫn ổ đĩa giữa Windows và Linux container.
- **Chia sẻ Volume `logs/`:** Container `api` mount `./logs:/app/logs` để ghi trực tiếp các file Parquet dự đoán ra máy host, giúp `drift.py` và Airflow của TV4 đọc được ngay.

---

## 3. Thiết kế FastAPI và các Endpoint

Mã nguồn nằm tại [src/api/app.py](../src/api/app.py):

| Endpoint | Phương thức | Chức năng |
|---|---|---|
| `/health` | `GET` | Kiểm tra trạng thái sống của service và trạng thái nạp model |
| `/model` | `GET` | Trả về `model_name`, `alias`, `version`, `model_type`, `threshold`, `run_id`, `loaded_at` |
| `/reload` | `POST` | Kiểm tra alias `@champion` trên MLflow và nạp lại nếu có phiên bản mới (hỗ trợ `?force=true`) |
| `/predict` | `POST` | Nhận 1 luồng (`dict`), danh sách luồng (`list[dict]`), hoặc `{"flows": [...]}` |
| `/predict/batch` | `POST` | Nhận một lô luồng mạng kèm `request_ids` và `timestamps` tùy chọn để phục vụ phát lại tốc độ cao |
| `/flush` | `POST` | Ép ghi ngay toàn bộ các bản ghi dự đoán còn trong bộ đệm xuống file Parquet |
| `/metrics` | `GET` | Xuất các chỉ số vận hành theo định dạng chuẩn của Prometheus |

**Xử lý hợp đồng dữ liệu và lỗi:**
- Đầu vào chỉ yêu cầu **70 cột đặc trưng gốc** trong `data/preprocessing/features.json` (2 cột cờ `fwd_win_missing` và `bwd_win_missing` được model `IDSModel` tự tính từ giá trị `-1`).
- Tự động ép kiểu các cột đầu vào sang `float64` trước khi đưa vào `mlflow.pyfunc` để tránh lỗi kiểm tra schema `int64` của MLflow.
- Thiếu cột đặc trưng, sai kiểu dữ liệu hoặc chứa `NaN`/`inf` → trả về **HTTP 422**.
- Chưa có model `@champion` trên MLflow → trả về **HTTP 503** (nếu đang chạy mà nạp lại bản mới thất bại thì giữ nguyên bản đang chạy trong RAM).

---

## 4. Ghi log dự đoán và nhãn trễ (Parquet)

Mã nguồn nằm tại [src/api/logger.py](../src/api/logger.py) và [src/api/replay.py](../src/api/replay.py):

1. **Log dự đoán (`logs/predictions/date=YYYY-MM-DD/part-<HH>h-<MMSS>-<seq>.parquet`):**
   - `PredictionLogger` gom kết quả dự đoán trong bộ nhớ đệm và chỉ ghi xuống đĩa khi đủ **1.000 dòng** (`LOG_FLUSH_MAX_ROWS=1000`), hoặc sau mỗi **60 giây** (`LOG_FLUSH_INTERVAL_SEC=60`), hoặc khi gọi `POST /flush`.
   - Các cột trong file Parquet: `request_id` (`string`), `timestamp` (`datetime64[us, UTC]`), `model_version` (`string`), các cột đặc trưng trong `features.json` (`float32`, giá trị gốc chưa scale), `score` (`float64`), `prediction` (`int8`), `latency_ms` (`float32`).
2. **Log nhãn thật đến trễ (`logs/labels/date=YYYY-MM-DD/batch-<id>.parquet`):**
   - Khi chạy `src.api.replay`, nhãn thật (`request_id`, `Label`, `y_binary`) của lô `k` chỉ được ghi ra đĩa sau khi đã gửi xong lô `k + delay_batches` (mặc định trễ 1 lô theo `monitoring.yaml`).

---

## 5. Chỉ số Prometheus và Dashboard Grafana

- **Cấu hình Prometheus:** [deploy/prometheus/prometheus.yml](../deploy/prometheus/prometheus.yml) thu thập metrics từ `api:8000/metrics` và `pushgateway:9091` (bật `honor_labels: true` để giữ nguyên nhãn `scenario` và `model_version` do TV4 gửi lên).
- **Cấu hình Grafana tự động (Provisioning):**
  - Datasource: [deploy/grafana/provisioning/datasources/datasource.yml](../deploy/grafana/provisioning/datasources/datasource.yml)
  - Dashboard: [deploy/grafana/dashboards/cic_ids_overview.json](../deploy/grafana/dashboards/cic_ids_overview.json) gồm 6 panel (Lưu lượng request, Độ trễ suy luận p50/p95, Tỉ lệ dự đoán Attack, Tỉ lệ cột Drift kèm vạch ngưỡng 0.3, Recall cộng dồn theo lớp tấn công, Phiên bản Model & Trạng thái Retrain) cùng Annotation tự động đánh dấu thời điểm đổi `model_version`.
  - Cảnh báo (Alert Rules): [deploy/grafana/provisioning/alerting/alerts.yml](../deploy/grafana/provisioning/alerting/alerts.yml) gồm cảnh báo **Đỏ** (`ids_retrain_trigger == 1`) và cảnh báo **Vàng** (`ids_dataset_drift == 1`).
