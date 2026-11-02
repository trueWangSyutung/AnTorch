#!/usr/bin/env python3
"""Sync repository docs into the GitBook source tree."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
DOCS = SITE / "docs"
ASSETS = DOCS / "assets"
IMAGES = ASSETS / "images"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for source, target in {
        "(docs/api.md)": "(api.md)",
        "(docs/quickstart.md)": "(quickstart.md)",
        "(docs/architecture.md)": "(architecture.md)",
        "(docs/jni.md)": "(jni.md)",
        "(docs/gpu.md)": "(gpu.md)",
        "(datasets/README.md)": "(datasets.md)",
    }.items():
        readme = readme.replace(source, target)
    readme = readme.replace("(site/docs/assets/images/antorch-mnist.png)", "(assets/images/antorch-mnist.png)")
    readme = readme.replace("(docs/assets/images/antorch-mnist.png)", "(assets/images/antorch-mnist.png)")
    write(DOCS / "README.md", readme)

    for name in ["api.md", "quickstart.md", "architecture.md", "jni.md", "gpu.md", "models.md", "technical-report.md"]:
        copy(ROOT / "docs" / name, DOCS / name)

    datasets = (ROOT / "datasets" / "README.md").read_text(encoding="utf-8")
    write(DOCS / "datasets.md", datasets)
    copy(ROOT / "CHANGELOG.md", DOCS / "changelog.md")
    print("synced GitBook docs")


if __name__ == "__main__":
    main()
