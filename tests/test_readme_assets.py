from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEMO_IMAGES = {
    defect: ROOT / "docs" / "assets" / "demo" / f"hazelnut_{defect}.png"
    for defect in ("crack", "hole", "print", "cut")
}


def test_readme_has_one_anonymous_demo_image_per_defect() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for defect, image_path in DEMO_IMAGES.items():
        relative = image_path.relative_to(ROOT).as_posix()
        assert relative in readme
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        with Image.open(image_path) as image:
            assert image.size == (1024, 1024)
            assert image.mode == "RGB"
            assert image.getexif() == {}
            assert not {"author", "artist", "copyright"} & {
                key.lower() for key in image.info
            }
