import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from policylens.pipeline import build_rag_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local RAG artifacts.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if local artifacts already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_rag_artifacts(force=args.force)
    print("RAG index is ready.")


if __name__ == "__main__":
    main()
