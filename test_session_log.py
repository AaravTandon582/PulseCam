"""Smoke test for privacy-preserving session result export."""
import json
import tempfile

import session_log


def main():
    result = {"fs": 29.97, "snr": 4.2, "dominance": 1.8}
    with tempfile.TemporaryDirectory() as directory:
        path = session_log.save(directory, 72.34, 30, result, "pos")
        with open(path, encoding="utf-8") as file:
            saved = json.load(file)
    assert saved["bpm"] == 72.3
    assert saved["measurement_seconds"] == 30.0
    assert saved["method"] == "pos"
    assert session_log.robust_bpm([72, 73, 71, 72, 140, 138]) == 72.5
    print("session log export and robust finalization passed")


if __name__ == "__main__":
    main()
