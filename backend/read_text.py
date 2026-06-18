import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from policylens.ingestion import *  # noqa: F401,F403
from policylens.ingestion import main


if __name__ == "__main__":
    main()
