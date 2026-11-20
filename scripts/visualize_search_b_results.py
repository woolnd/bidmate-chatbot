"""최신 Search-B 산출물을 생성하는 호환 실행 진입점."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts" / "finalize_search_b.py"

raise SystemExit(
    subprocess.call(
        [sys.executable, str(FINALIZER)],
        cwd=ROOT,
    )
)
