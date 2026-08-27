from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "demo_assets" / "mvtec_ad"
HAZELNUT_ROOT = ASSET_ROOT / "hazelnut"

EXPECTED_HASHES = {
    "hazelnut/train/good/000.png": "5e28a714fa36ef5198c683058b607435b792974cdd23f3e0810b887dbdfe7112",
    "hazelnut/train/good/001.png": "612e2884069ce4ca26c96b9e885d54cee1cf3f16256c83a74b985966c4098c4f",
    "hazelnut/train/good/002.png": "96ec0d30fb80620a0c3dc09173a856cda9681a24c383866272a04f05e360c2d2",
    "hazelnut/train/good/003.png": "7c20b778f1226beb943485c0489373a8185fa5debbc8c371398e2d5762189cf4",
    "hazelnut/train/good/004.png": "de53dc392f6cdae2bb25e325de6c159af78552529fc2bdb35a0b8d7d7466e2f3",
    "hazelnut/train/good/005.png": "90a3f7e86066c8558d81b2c2f71ecbb7ad38de78d195deddbb3d78f0348dbb4d",
    "hazelnut/train/good/006.png": "0cfed0af8c0d3b69101e01849aea21ddd791a55e1f032c1a235de353190d47fc",
    "hazelnut/train/good/007.png": "b2679a80a85894a7a7bb465a7fb53f5f03d40d5b5b2fe8e0e516a4624802abb7",
    "hazelnut/train/good/008.png": "7cc3bc930ceca4b58cc45050dd4aed66e42dbc7270f90a275b692adea1031c2b",
    "hazelnut/train/good/009.png": "1502087011ec6a5ec6e089e64506526f1f46fa9e852aba88b2add9b02f99f0ec",
    "hazelnut/train/good/010.png": "4b3cc485775daaf8512305aae869799b6f5659833a4b1176df181c9ff7d9d102",
    "hazelnut/train/good/011.png": "ac46ac9b51f6af9eaf033f03615270ef9fb59608dbfbd8a386871be3a04f0303",
    "hazelnut/train/good/012.png": "ac46f6aa2cc8618d5d04b468390a5c9cd2f331dd3cbbfd80f69b9013f05eef0d",
    "hazelnut/train/good/013.png": "511fb2e58ff750e69aea2aab59b95301e409ca923645e08452e06614b12688a4",
    "hazelnut/train/good/014.png": "7924bcfefa9736647abdf63409ce279d6df11a8e03b0aaf5bc0f9fb4db5ef601",
    "hazelnut/test/crack/000.png": "e26b54282b6b11286760905ca2880af594561b606dfa0643859d6ee6027cf8c2",
    "hazelnut/test/hole/000.png": "b9a1a102db263079fec8bc9836f182253dced9ca178957b7e9bef05cba3312c2",
    "hazelnut/test/print/000.png": "63245ac91c1f83c97e081bbcfde5bb9751b008a994c201d9f18dcd6db3552c87",
    "hazelnut/test/cut/000.png": "901bf3b3ddef6e7d3592b00cfd0f5c512b745ef61e7837ed4e283da2f99c1b8a",
    "hazelnut/ground_truth/crack/000_mask.png": "299566b57b84b9d466f4f7f50a8210e0bd0ff2a4ac5922f456d00cd9169cfbb4",
    "hazelnut/ground_truth/hole/000_mask.png": "0db752dd6620c7c7493fe8b150e2a706d4c851a15a27c919e545552850e85018",
    "hazelnut/ground_truth/print/000_mask.png": "2ad368e89aa0f9e28dd3df9a5581ab9815916447abbb5e13ee0405f666057870",
    "hazelnut/ground_truth/cut/000_mask.png": "5dc3b2dc619d9e0d15fed32cf39b3bc8b2ba425964a00e947745c632ccf73c5b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_demo_subset_is_exact_and_auditable() -> None:
    actual = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in HAZELNUT_ROOT.rglob("*.png")
    }
    assert actual == set(EXPECTED_HASHES)
    for relative, expected_hash in EXPECTED_HASHES.items():
        path = ASSET_ROOT / relative
        assert path.stat().st_size > 0
        assert sha256(path) == expected_hash
        with Image.open(path) as image:
            assert image.size == (1024, 1024)
            expected_mode = "L" if "ground_truth" in relative else "RGB"
            assert image.mode == expected_mode

    manifest = (ASSET_ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    assert manifest == [f"{digest}  {relative}" for relative, digest in EXPECTED_HASHES.items()]
