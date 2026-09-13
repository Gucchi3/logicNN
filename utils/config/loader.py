"""設定JSONを読み込み、利用者向けエラーへ変換する。"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from utils.exceptions import LogicNNError

from .schema import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> AppConfig:
    """設定JSONを厳格に検証し、設定内のパスをプロジェクトルート基準で解決する。"""
    config_path = Path(path).expanduser().resolve()
    try:
        source = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise LogicNNError(
            "設定ファイルを読み込めません",
            detail=f"対象: {config_path}\n原因: {error}",
            hint="存在するUTF-8形式の設定JSONを--configで指定してください",
        ) from error

    try:
        config = AppConfig.model_validate_json(source)
    except ValidationError as error:
        raise LogicNNError(
            "設定ファイルの内容が不正です",
            detail=_format_validation_error(config_path, error),
            hint="表示された設定項目を修正し、もう一度実行してください",
        ) from error
    return _resolve_config_paths(config)


def _format_validation_error(config_path: Path, error: ValidationError) -> str:
    """Pydanticの全違反を設定パスと入力値を含む読みやすい文字列へ変換する。"""
    violations = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "JSON"
        violations.append(f"{location}: {item['msg']} (入力値: {item.get('input')!r})")
    return f"対象: {config_path}\n" + "\n".join(violations)


def _resolve_config_paths(config: AppConfig) -> AppConfig:
    """設定に含まれる相対パスをプロジェクトルート基準の絶対パスへ変換する。"""
    initial_checkpoint_path = _resolve_path(config.run.initial_checkpoint_path) if config.run.initial_checkpoint_path is not None else None
    run  = config.run.model_copy(update={"log_dir": _resolve_path(config.run.log_dir), "initial_checkpoint_path": initial_checkpoint_path})
    data = config.data.model_copy(update={"root": _resolve_path(config.data.root)})
    return config.model_copy(update={"run": run, "data": data})


def _resolve_path(path: Path) -> Path:
    """絶対パスは維持し、相対パスだけをプロジェクトルートから解決する。"""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
