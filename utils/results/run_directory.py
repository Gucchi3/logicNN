"""既存成果物を上書きしない実行ディレクトリを作成する。"""

from datetime import datetime
from pathlib import Path

from utils.exceptions import LogicNNError


def create_run_directory(log_dir: Path) -> Path:
    """現在のローカル日時と衝突時の連番で新しい実行ディレクトリを確保する。"""
    log_dir = Path(log_dir)
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        while True:
            candidate = log_dir / (stamp if index == 0 else f"{stamp}_{index:02d}")
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                index += 1
    except OSError as error:
        raise LogicNNError("実行ディレクトリを作成できません", detail=f"{log_dir}: {error}", hint="保存先と書込権限を確認してください") from error
