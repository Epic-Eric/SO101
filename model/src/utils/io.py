import json
from pathlib import Path
from typing import List

import cv2

from ..interfaces.teleop import TeleopSample, TeleopSessionResult


def write_frame(path: str | Path, image_bgr) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), image_bgr)


def append_actions_jsonl(path: str | Path, timestamp: float, action: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", buffering=1) as fp:
        fp.write(json.dumps({"t": timestamp, "action": action}) + "\n")


def save_session(result: TeleopSessionResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions_path = out / "actions.jsonl"

    for i, sample in enumerate(result.samples):
        frame = sample.observation.get("front")
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            write_frame(out / f"frame_{i:06d}.jpg", bgr)
        append_actions_jsonl(actions_path, sample.timestamp, sample.action)


def load_session(out_dir: str | Path) -> TeleopSessionResult:
    # Minimal loader: actions only; frames can be read by globbing
    out = Path(out_dir)
    actions_path = out / "actions.jsonl"
    samples: List[TeleopSample] = []
    if actions_path.exists():
        with actions_path.open() as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                    samples.append(
                        TeleopSample(timestamp=rec.get("t", 0.0), action=rec.get("action", {}), observation={})
                    )
                except Exception:
                    continue
    return TeleopSessionResult(samples=samples, metadata={"loaded": True}, output_dir=str(out))
