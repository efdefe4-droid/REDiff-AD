from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IDENTIFIERS = (
    b"efdefe4" + b"-droid",
    b"tsmc" + b"-tsai",
    b"/home/" + b"tsmc" + b"-tsai",
)


def submission_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    candidates = [ROOT / path.decode("utf-8") for path in completed.stdout.split(b"\0") if path]
    return [path for path in candidates if path.is_file()]


def test_tracked_artifacts_do_not_disclose_private_identity() -> None:
    disclosures: list[str] = []
    for path in submission_files():
        data = path.read_bytes()
        for identifier in FORBIDDEN_IDENTIFIERS:
            if identifier.lower() in data.lower():
                disclosures.append(f"{path.relative_to(ROOT)}: {identifier.decode('utf-8')}")
    assert not disclosures, "private identifiers found:\n" + "\n".join(disclosures)
