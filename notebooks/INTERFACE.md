# Giao tiếp TV3 (Serving & Infra) ↔ TV4 (Monitoring & Orchestration)

**Trạng thái:** đề xuất, chờ TV3 và nhóm chốt. Khi một mục đã chốt, đổi `[ ]` thành `[x]` và ghi lại mọi thay đổi ngay trong file này.

**Mục đích:** TV3 làm API và Docker, TV4 làm giám sát và retrain, hai bên làm **song song**. Hai bên chỉ cần khớp nhau ở 4 điểm dưới đây.

```
                    ┌──────────── MLflow server (1) ────────────┐
                    │  Registry: cic-ids-binary @champion       │
                    └──────▲──────────────────────────▲─────────┘
            load @champion │                          │ đăng ký / promote model mới
                           │                          │
  request ──► API FastAPI (TV3) ──► log prediction (2) ──► drift.py / simulate.py (TV4)
                  │                                              │
                  │ /metrics                                     │ push chỉ số drift
                  ▼                                              ▼
              Prometheus (4) ◄──────────── Pushgateway ◄─────────┘
                  │
                  ▼
               Grafana
```

---

## 1. MLflow server dùng chung (ưu tiên cao nhất)

Hiện mỗi người dùng `sqlite:///mlruns/mlflow.db` trên máy mình, nên API không thấy model do TV2 hoặc TV4 đăng ký.

- [ ] TV3 thêm service `mlflow` vào `docker-compose.yml`:
  - backend store: Postgres hoặc sqlite trên volume;
  - artifact store: volume hoặc MinIO;
  - cổng `5000`.
- [ ] Mọi thành phần (train, API, monitoring, Airflow) trỏ tới server này qua biến môi trường `MLFLOW_TRACKING_URI=http://mlflow:5000`. `models.yaml` đã hỗ trợ ghi đè bằng biến này nên **không cần sửa code**.

## 2. Model và cách nạp

Đã có sẵn từ TV2, xem `src/models/pyfunc.py` và `notebooks/MODEL.md`.

| Mục | Giá trị |
|---|---|
| Tên model trong registry | `cic-ids-binary` |
| Alias | `@champion`: bản API dùng · `@challenger`: bản mới chờ so sánh · `@previous`: bản để rollback |
| Nạp model | `mlflow.pyfunc.load_model("models:/cic-ids-binary@champion")` |
| Đầu vào | DataFrame gồm 70 cột đặc trưng theo `data/preprocessing/features.json`. Không cần gửi 2 cột cờ `fwd_win_missing` / `bwd_win_missing`, model tự tính từ giá trị −1. Thiếu cột hoặc có NaN/inf thì model báo `ValueError`, nên API trả về **422** |
| Đầu ra | `score` (float), `prediction` (0/1), `label` (`"Benign"` / `"Attack"`) |
| Ngưỡng | Nằm sẵn trong model, xem `model.metadata.metadata["threshold"]`. **API không tự đặt ngưỡng**, để API và giám sát luôn khớp nhau |
| Phiên bản | Lấy qua `MlflowClient().get_model_version_by_alias(...)`. API ghi phiên bản này vào log (mục 3) và hiển thị ở `GET /model` |

- [ ] **Đổi model không cần khởi động lại API.** Mỗi 60 giây, API kiểm tra alias `@champion` trỏ tới phiên bản nào. Nếu phiên bản khác bản đang chạy thì API nạp lại model. Có thể thêm `POST /reload` để Airflow gọi ngay sau khi promote.

## 3. Log prediction (API ghi, TV4 đọc)

Thành phần giám sát dùng log này **thay cho `production_sim.parquet`**. Bên giám sát chỉ cần đổi nguồn dữ liệu, còn cách cắt lô và kiểm tra giữ nguyên.

- [ ] **Vị trí:** volume chung `logs/predictions/`, mỗi ngày (UTC) một thư mục: `logs/predictions/date=YYYY-MM-DD/part-<giờ>.parquet`.
- [ ] **Định dạng:** Parquet, ghi theo đợt (ví dụ mỗi 1.000 request hoặc mỗi 60 giây, tùy điều kiện nào đến trước). Không ghi từng dòng vì quá chậm.
- [ ] **Các cột:**

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `request_id` | string (UUID) | Để ghép với nhãn đến sau |
| `timestamp` | timestamp (UTC) | Thời điểm nhận request; dùng để cắt lô theo thời gian |
| `model_version` | string | Phiên bản model đã dự đoán; dùng để tách số liệu trước và sau khi đổi model |
| 70 cột đặc trưng | float32 | **Đúng tên** trong `features.json`, giá trị gốc (chưa scale) |
| `score` | float64 | Đầu ra của model |
| `prediction` | int8 | 0/1 |
| `latency_ms` | float32 | Không bắt buộc |

- [ ] **Nhãn** (giả lập việc analyst xác nhận sau): file riêng `logs/labels/date=YYYY-MM-DD/*.parquet` gồm `request_id, Label`. Khi demo, script phát lại dữ liệu gửi request lên API và ghi nhãn thật vào đây, chậm một lô so với request.

## 4. Chỉ số trên Prometheus / Grafana

**API (TV3)** expose qua `GET /metrics` bằng `prometheus-client`:

| Tên | Loại | Nhãn |
|---|---|---|
| `ids_requests_total` | Counter | `status` (ok / error) |
| `ids_predictions_total` | Counter | `label` (Benign / Attack), `model_version` |
| `ids_score` | Histogram (bucket 0.1 … 1.0) | `model_version` |
| `ids_inference_latency_seconds` | Histogram | – |
| `ids_model_info` | Gauge (= 1) | `version`, `model_type`, `threshold` |

**Giám sát (TV4)** push qua **Pushgateway**:
- [ ] TV3 thêm service `pushgateway` vào compose, cổng `9091`.
- Mỗi lô push một lần; job `ids_monitoring`; nhãn chung `scenario`, `model_version`.

| Tên | Loại | Ý nghĩa |
|---|---|---|
| `ids_drift_share` | Gauge | Tỉ lệ cột đặc trưng drift của lô gần nhất |
| `ids_dataset_drift` | Gauge (0/1) | Lô gần nhất có bị coi là drift không (≥ `drift_share` số cột) |
| `ids_score_drift` | Gauge | Khoảng cách phân phối `score` so với reference |
| `ids_alert_rate` | Gauge | Tỉ lệ dự đoán Attack của lô |
| `ids_recall` | Gauge, nhãn `label` | Recall cộng dồn theo lớp tấn công, khi đã có nhãn |
| `ids_fpr` | Gauge | FPR của lô đã có nhãn gần nhất |
| `ids_retrain_trigger` | Gauge (0/1) | Điều kiện retrain đang kích hoạt |
| `ids_retrain_total` | Counter | Số lần đã retrain hoặc promote |

- [ ] **Grafana (TV3 dựng, TV4 góp ý):**
  - Panel lưu lượng và latency.
  - Panel tỉ lệ Attack theo thời gian.
  - Panel drift (`ids_drift_share` kèm đường ngưỡng).
  - Panel recall theo lớp.
  - Đánh dấu thời điểm đổi `model_version`.
- [ ] **Cảnh báo Grafana:**
  - `ids_retrain_trigger == 1` → cảnh báo đỏ.
  - `ids_dataset_drift == 1` hoặc `ids_alert_rate` vượt 10 lần mốc val → cảnh báo vàng. Chỉ để báo, **không tự retrain** (xem Kịch bản 2).

---

## Phân việc và thứ tự

| Bước | Ai | Việc | Phụ thuộc |
|---|---|---|---|
| 1 | TV3 | `docker-compose.yml`: `mlflow`, `pushgateway`, `prometheus`, `grafana` | – |
| 2 | TV3 | API `/predict`, `/model`, `/metrics`, `/reload` và ghi log theo mục 3 | Model `@champion` (đã có) |
| 2 | TV4 | `simulate.py`: vòng retrain offline trên `production_sim` | – (song song với TV3) |
| 3 | TV4 | `drift.py` đọc thêm nguồn `logs/predictions` và push lên Pushgateway | Bước 1, 2 |
| 4 | TV4 | DAG Airflow: giám sát định kỳ → retrain → so sánh → promote → `POST /reload` | Bước 1–3 |
| 5 | Cả hai | Demo end-to-end: script phát lại `production_sim` gửi tới API | Tất cả |

## Câu hỏi cần TV3 trả lời

1. Dùng Postgres cho MLflow hay sqlite trên volume là đủ cho phạm vi bài tập lớn?
2. API nhận **một luồng mỗi request** hay **nhiều luồng mỗi request**? Khi demo phát lại hàng triệu dòng thì nên có `POST /predict/batch`.
3. API tự nạp lại model định kỳ (mỗi 60 giây) hay chỉ nạp khi gọi `/reload`?
