"""FastAPI phục vụ mô hình phát hiện xâm nhập mạng (TV3 - Serving & Infra).

Hợp đồng giao tiếp theo notebooks/INTERFACE.md:
  - Nạp mô hình từ MLflow Model Registry: models:/cic-ids-binary@champion
  - Tự động kiểm tra phiên bản @champion định kỳ (mặc định 60s) và hỗ trợ POST /reload
  - Ghi log dự đoán ra Parquet theo đợt vào logs/predictions/date=YYYY-MM-DD/part-*.parquet
  - Expose chỉ số Prometheus tại GET /metrics
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import Body, FastAPI, HTTPException, Response
import mlflow
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
import yaml

from src.api.logger import PredictionLogger
from src.models import data as D
from src.models import registry as R
from src.models.pyfunc import WINDOW_FLAGS

# ---------------------------------------------------------------------------
# Chỉ số Prometheus (Mục 4 - INTERFACE.md)
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "ids_requests_total",
    "Tổng số request gọi vào API dự đoán",
    ["status"],
    registry=REGISTRY,
)
PREDICTIONS_TOTAL = Counter(
    "ids_predictions_total",
    "Tổng số luồng mạng đã dự đoán theo nhãn và phiên bản model",
    ["label", "model_version"],
    registry=REGISTRY,
)
SCORE_HISTOGRAM = Histogram(
    "ids_score",
    "Phân phối điểm dự đoán Attack (score)",
    ["model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)
INFERENCE_LATENCY = Histogram(
    "ids_inference_latency_seconds",
    "Thời gian suy luận của mô hình (giây)",
    registry=REGISTRY,
)
MODEL_INFO = Gauge(
    "ids_model_info",
    "Thông tin phiên bản mô hình đang phục vụ (= 1)",
    ["version", "model_type", "threshold"],
    registry=REGISTRY,
)
MODEL_VERSION = Gauge(
    "ids_model_version",
    "Phiên bản số của mô hình đang phục vụ (dùng để phát hiện thời điểm đổi model)",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Quản lý nạp và tự động đổi mô hình @champion
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg["registry"]["model_name"]
        self.alias = cfg["registry"]["champion_alias"]
        self.all_features = D.load_features(cfg["data"]["artifacts_dir"])
        self.required_features = [c for c in self.all_features if c not in WINDOW_FLAGS]

        self._lock = threading.RLock()
        self.model = None
        self.version: Optional[str] = None
        self.model_type: str = "unknown"
        self.threshold: float = 0.5
        self.run_id: Optional[str] = None
        self.loaded_at: Optional[str] = None

    def check_and_reload(self, force: bool = False) -> Dict[str, Any]:
        """Kiểm tra alias @champion trên MLflow; nếu khác bản đang chạy (hoặc force=True) thì nạp lại."""
        R.setup(self.cfg)
        mv = R.get_version(self.model_name, self.alias)
        if mv is None:
            return {
                "reloaded": False,
                "previous_version": self.version,
                "current_version": self.version,
                "detail": f"Chưa có mô hình '{self.model_name}@{self.alias}' trên MLflow Registry",
            }

        target_version = str(mv.version)
        if not force and self.model is not None and self.version == target_version:
            return {
                "reloaded": False,
                "previous_version": self.version,
                "current_version": self.version,
                "detail": "Phiên bản @champion không thay đổi",
            }

        uri = R.model_uri(self.model_name, self.alias)
        loaded = mlflow.pyfunc.load_model(uri)
        meta = getattr(loaded.metadata, "metadata", None) or {}
        unwrapped = None
        try:
            unwrapped = loaded.unwrap_python_model()
        except Exception:
            pass

        model_type = str(
            meta.get("model_type")
            or getattr(unwrapped, "model_type", None)
            or (mv.tags or {}).get("model_type", "unknown")
        )
        threshold = float(
            meta.get("threshold")
            if meta.get("threshold") is not None
            else getattr(unwrapped, "threshold", 0.5)
        )
        run_id = getattr(loaded.metadata, "run_id", None) or mv.run_id
        loaded_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            prev_version = self.version
            self.model = loaded
            self.version = target_version
            self.model_type = model_type
            self.threshold = threshold
            self.run_id = run_id
            self.loaded_at = loaded_at

            MODEL_INFO.clear()
            MODEL_INFO.labels(
                version=self.version,
                model_type=self.model_type,
                threshold=f"{self.threshold:.6f}",
            ).set(1)
            try:
                MODEL_VERSION.set(float(self.version))
            except (TypeError, ValueError):
                pass

        return {
            "reloaded": True,
            "previous_version": prev_version,
            "current_version": self.version,
            "model_type": self.model_type,
            "threshold": self.threshold,
            "run_id": self.run_id,
            "loaded_at": self.loaded_at,
        }

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "model_name": self.model_name,
                "alias": self.alias,
                "loaded": self.model is not None,
                "version": self.version,
                "model_type": self.model_type if self.model is not None else None,
                "threshold": self.threshold if self.model is not None else None,
                "run_id": self.run_id,
                "loaded_at": self.loaded_at,
                "required_features_count": len(self.required_features),
            }

    def predict_df(self, raw_df: pd.DataFrame):
        """Kiểm tra đầu vào, ép sang float64 và gọi pyfunc model.predict()."""
        with self._lock:
            model = self.model
            version = self.version
            threshold = self.threshold
            model_type = self.model_type

        if model is None:
            # Thử nạp lần nữa phòng trường hợp model mới được đăng ký sau khi API bật
            try:
                self.check_and_reload()
            except Exception:
                pass
            with self._lock:
                model = self.model
                version = self.version
                threshold = self.threshold
                model_type = self.model_type
            if model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Mô hình '{self.model_name}@{self.alias}' chưa sẵn sàng trên MLflow Registry.",
                )

        if len(raw_df) == 0:
            raise ValueError("Danh sách luồng dự đoán không được rỗng.")

        missing = [c for c in self.required_features if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Thiếu {len(missing)} cột đặc trưng: {missing}")

        # Ép kiểu float64 theo đúng ModelSignature của pyfunc (tránh lỗi int64 của MLflow)
        cols_present = [c for c in self.all_features if c in raw_df.columns]
        try:
            X_f64 = raw_df[cols_present].astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Giá trị đặc trưng không hợp lệ (phải là số thực): {exc}") from exc

        bad_mask = ~np.isfinite(X_f64.to_numpy()).all(axis=1)
        if bad_mask.any():
            first_bad = int(np.flatnonzero(bad_mask)[0])
            raise ValueError(f"{int(bad_mask.sum())} dòng có NaN/inf (dòng đầu tiên: {first_bad})")

        t0 = time.perf_counter()
        out_df = model.predict(X_f64)
        elapsed_sec = time.perf_counter() - t0
        return X_f64, out_df, version, threshold, model_type, elapsed_sec


# ---------------------------------------------------------------------------
# Khởi tạo ứng dụng FastAPI & Vòng đời (Lifespan)
# ---------------------------------------------------------------------------
class BatchPredictRequest(BaseModel):
    flows: List[Dict[str, Any]]
    request_ids: Optional[List[str]] = None
    timestamps: Optional[List[str]] = None


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = path or os.environ.get("MODELS_CONFIG", "models.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_app(cfg: Optional[dict] = None, log_dir: Optional[str] = None,
               reload_interval_sec: Optional[float] = None,
               log_flush_max_rows: Optional[int] = None,
               log_flush_interval_sec: Optional[float] = None) -> FastAPI:
    cfg = cfg or load_config()
    log_dir = log_dir or os.environ.get("PREDICTION_LOG_DIR", "logs/predictions")
    reload_sec = (
        reload_interval_sec
        if reload_interval_sec is not None
        else float(os.environ.get("MODEL_RELOAD_INTERVAL_SEC", "60"))
    )
    flush_rows = (
        log_flush_max_rows
        if log_flush_max_rows is not None
        else int(os.environ.get("LOG_FLUSH_MAX_ROWS", "1000"))
    )
    flush_sec = (
        log_flush_interval_sec
        if log_flush_interval_sec is not None
        else float(os.environ.get("LOG_FLUSH_INTERVAL_SEC", "60"))
    )

    manager = ModelManager(cfg)
    pred_logger = PredictionLogger(
        log_dir=log_dir,
        features=manager.all_features,
        max_rows=flush_rows,
        flush_interval_sec=flush_sec,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            manager.check_and_reload()
        except Exception:
            pass

        stop_event = asyncio.Event()

        async def _background_worker():
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=min(reload_sec, flush_sec, 10.0))
                except asyncio.TimeoutError:
                    pass
                if stop_event.is_set():
                    break
                try:
                    await asyncio.to_thread(manager.check_and_reload, False)
                except Exception:
                    pass
                try:
                    await asyncio.to_thread(pred_logger.flush_if_due)
                except Exception:
                    pass

        task = asyncio.create_task(_background_worker())
        try:
            yield
        finally:
            stop_event.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            pred_logger.flush()

    app = FastAPI(
        title="CIC-IDS-2018 Intrusion Detection API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_manager = manager
    app.state.prediction_logger = pred_logger

    def _run_inference(flows: List[Dict[str, Any]], request_ids: Optional[List[str]] = None,
                       timestamps: Optional[List[str]] = None, single_item: bool = False):
        if request_ids is not None and len(request_ids) != len(flows):
            REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=422, detail="Độ dài request_ids không khớp với số lượng flows.")
        if timestamps is not None and len(timestamps) != len(flows):
            REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=422, detail="Độ dài timestamps không khớp với số lượng flows.")

        raw_df = pd.DataFrame(flows)
        try:
            X_f64, out_df, version, threshold, model_type, elapsed_sec = manager.predict_df(raw_df)
        except HTTPException:
            REQUESTS_TOTAL.labels(status="error").inc()
            raise
        except (ValueError, TypeError, KeyError, MlflowException) as exc:
            REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        latency_ms = elapsed_sec * 1000.0
        INFERENCE_LATENCY.observe(elapsed_sec)
        REQUESTS_TOTAL.labels(status="ok").inc()

        scores_arr = out_df["score"].to_numpy(dtype=np.float64)
        preds_arr = out_df["prediction"].to_numpy(dtype=np.int64)
        labels_arr = out_df["label"].tolist()

        n_attack = int((preds_arr == 1).sum())
        n_benign = len(preds_arr) - n_attack
        if n_benign:
            PREDICTIONS_TOTAL.labels(label="Benign", model_version=str(version)).inc(n_benign)
        if n_attack:
            PREDICTIONS_TOTAL.labels(label="Attack", model_version=str(version)).inc(n_attack)

        hist_child = SCORE_HISTOGRAM.labels(model_version=str(version))
        for s in scores_arr:
            hist_child.observe(float(s))

        req_ids = pred_logger.record(
            features_df=X_f64,
            out_df=out_df,
            model_version=str(version),
            latency_ms=latency_ms,
            request_ids=request_ids,
            timestamps=timestamps,
        )

        items = [
            {
                "request_id": req_ids[i],
                "score": float(scores_arr[i]),
                "prediction": int(preds_arr[i]),
                "label": str(labels_arr[i]),
                "model_version": str(version),
            }
            for i in range(len(flows))
        ]
        if single_item:
            return {
                **items[0],
                "threshold": threshold,
                "latency_ms": round(latency_ms, 3),
            }
        return {
            "model_version": str(version),
            "model_type": model_type,
            "threshold": threshold,
            "count": len(items),
            "attack_count": n_attack,
            "latency_ms": round(latency_ms, 3),
            "predictions": items,
        }

    @app.get("/health")
    def health():
        info = manager.info()
        return {"status": "ok", "model_loaded": info["loaded"], "model_version": info["version"]}

    @app.get("/model")
    def get_model():
        info = manager.info()
        if not info["loaded"]:
            try:
                manager.check_and_reload()
                info = manager.info()
            except Exception:
                pass
        return info

    @app.post("/reload")
    def reload_model(force: bool = False):
        try:
            return manager.check_and_reload(force=force)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Không nạp được model @champion: {exc}") from exc

    @app.post("/predict")
    def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...)):
        """Nhận 1 luồng (dict các đặc trưng), danh sách luồng (list[dict]), hoặc {"flows": [...]}."""
        if isinstance(payload, list):
            return _run_inference(payload, single_item=False)
        if "flows" in payload and isinstance(payload["flows"], list):
            return _run_inference(
                payload["flows"],
                request_ids=payload.get("request_ids"),
                timestamps=payload.get("timestamps"),
                single_item=False,
            )
        req_id = payload.get("request_id")
        ts = payload.get("timestamp")
        return _run_inference(
            [payload],
            request_ids=[str(req_id)] if req_id is not None else None,
            timestamps=[str(ts)] if ts is not None else None,
            single_item=True,
        )

    @app.post("/predict/batch")
    def predict_batch(payload: Union[BatchPredictRequest, List[Dict[str, Any]]] = Body(...)):
        """Nhận một lô nhiều luồng mạng để dự đoán tốc độ cao khi phát lại dữ liệu."""
        if isinstance(payload, list):
            return _run_inference(payload, single_item=False)
        return _run_inference(
            payload.flows,
            request_ids=payload.request_ids,
            timestamps=payload.timestamps,
            single_item=False,
        )

    @app.post("/flush")
    def flush_logs():
        """Ghi ngay các bản ghi dự đoán còn trong bộ đệm xuống đĩa Parquet."""
        flushed = pred_logger.flush()
        return {"flushed_rows": flushed}

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
