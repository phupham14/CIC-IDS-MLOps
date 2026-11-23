"""Đẩy chỉ số giám sát lên Prometheus Pushgateway (INTERFACE.md, mục 4).

Mỗi lần push ghi đè toàn bộ nhóm (job, grouping key) trên Pushgateway, nên registry ở đây giữ giá trị cuối cùng
của mọi gauge: lô chưa có nhãn vẫn đẩy lại recall/FPR/trigger của lần đánh giá gần nhất.
Push lỗi (Pushgateway chưa bật) chỉ ghi cảnh báo, không dừng giám sát.
"""
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.models import metrics as M
from src.models.train import log

GAUGES = {                                        # tên -> (mô tả, khóa trong dòng kết quả của một lô)
    "ids_drift_share": ("Tỉ lệ cột đặc trưng drift của lô gần nhất", "drift_share"),
    "ids_dataset_drift": ("Lô gần nhất bị coi là drift (0/1)", "dataset_drift"),
    "ids_score_drift": ("Khoảng cách phân phối score so với reference", "score_drift"),
    "ids_alert_rate": ("Tỉ lệ dự đoán Attack của lô", "alert_rate"),
    "ids_monitor_batch": ("Số thứ tự lô vừa giám sát", "batch"),
}
LABELED = {                                       # chỉ cập nhật khi nhãn của một lô đến
    "ids_fpr": ("FPR của lô đã có nhãn gần nhất", "fpr"),
    "ids_retrain_trigger": ("Điều kiện retrain đang kích hoạt (0/1)", "retrain_trigger"),
    "ids_monitor_labeled_batch": ("Số thứ tự lô có nhãn gần nhất", "labeled_batch"),
}


class Pusher:
    """push(row) sau mỗi lô; row là một dòng kết quả của drift.monitor hoặc live.LiveMonitor."""

    def __init__(self, url, job="ids_monitoring", grouping_key=None, pause_sec=0.0, push_fn=push_to_gateway,
                 timeout_sec=3.0, retry_sec=30.0):
        self.url, self.job, self.grouping_key = url, job, dict(grouping_key or {})
        self.pause_sec, self.push_fn = float(pause_sec), push_fn
        self.timeout_sec, self.retry_sec = float(timeout_sec), float(retry_sec)
        self._skip_until = 0.0                    # sau một lần lỗi, tạm không push để không làm chậm giám sát
        self.registry = CollectorRegistry()
        self.gauges = {name: Gauge(name, doc, ["model_version"], registry=self.registry)
                       for name, (doc, _) in {**GAUGES, **LABELED}.items()}
        self.recall = Gauge("ids_recall", "Recall cộng dồn theo lớp tấn công (từ khi mô hình hiện hành chạy)",
                            ["model_version", "label"], registry=self.registry)
        self.version = None
        self.failures = 0

    def update(self, row, model_version):
        v = str(model_version)
        if v != self.version:                     # đổi mô hình: bỏ chuỗi số liệu của phiên bản cũ
            for g in (*self.gauges.values(), self.recall):
                g.clear()
            self.version = v
        for name, (_, key) in GAUGES.items():
            if key in row:
                self.gauges[name].labels(v).set(float(row[key]))
        if "labeled_batch" in row:
            for name, (_, key) in LABELED.items():
                if key in row:
                    self.gauges[name].labels(v).set(float(row[key]))
            for lab, r in (row.get("cum_recall") or {}).items():
                if lab != M.BENIGN:
                    self.recall.labels(v, lab).set(float(r))

    def push(self, row, model_version):
        """Trả về True nếu push thành công. Lần push sau khi Pushgateway hoạt động lại vẫn mang giá trị mới nhất."""
        self.update(row, model_version)
        ok = False
        if time.monotonic() >= self._skip_until:
            try:
                self.push_fn(self.url, job=self.job, registry=self.registry, grouping_key=self.grouping_key,
                             timeout=self.timeout_sec)
                ok = True
            except Exception as exc:              # noqa: BLE001 (Pushgateway tắt không được làm dừng giám sát)
                self.failures += 1
                self._skip_until = time.monotonic() + self.retry_sec
                log(f"Không push được lên {self.url} (lần lỗi thứ {self.failures}), thử lại sau "
                    f"{self.retry_sec:.0f}s: {exc}")
        if ok and self.pause_sec:                        # để Prometheus kịp scrape từng lô khi phát lại nhanh
            time.sleep(self.pause_sec)
        return ok
