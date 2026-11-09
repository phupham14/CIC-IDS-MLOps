"""Kết nối MLflow và thao tác Model Registry.

Alias: @challenger (bản mới), @champion (bản API dùng), @previous (champion trước, để rollback).
"""
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def setup(cfg, experiment=None):
    """Đặt tracking URI và experiment."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    name = experiment or cfg["mlflow"]["experiment"]
    mlflow.set_tracking_uri(uri)
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        if mlflow.get_experiment_by_name(name) is None:
            root = Path(cfg["mlflow"]["artifact_root"]) / name
            mlflow.create_experiment(name, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(name)
    return uri


def model_uri(name, alias_or_version):
    s = str(alias_or_version)
    return f"models:/{name}/{s}" if s.isdigit() else f"models:/{name}@{s}"


def get_version(name, alias_or_version):
    """Lấy phiên bản theo số hoặc alias; None nếu chưa có."""
    client = MlflowClient()
    try:
        s = str(alias_or_version)
        return client.get_model_version(name, s) if s.isdigit() else client.get_model_version_by_alias(name, s)
    except MlflowException:
        return None


def register(model_uri_, name, alias, tags=None):
    mv = mlflow.register_model(model_uri_, name, tags=tags)
    MlflowClient().set_registered_model_alias(name, alias, str(mv.version))
    return mv


def promote(name, version, reg_cfg, tags=None):
    """Chuyển @champion sang version, champion cũ thành @previous."""
    client = MlflowClient()
    version = str(version)
    old = get_version(name, reg_cfg["champion_alias"])
    old_v = str(old.version) if old is not None else None
    if old_v is not None and old_v != version:
        client.set_registered_model_alias(name, reg_cfg["previous_alias"], old_v)
    client.set_registered_model_alias(name, reg_cfg["champion_alias"], version)
    mv = get_version(name, version)
    if reg_cfg["challenger_alias"] in (mv.aliases or []):
        client.delete_registered_model_alias(name, reg_cfg["challenger_alias"])
    for k, v in (tags or {}).items():
        client.set_model_version_tag(name, version, k, str(v))
    return old_v
