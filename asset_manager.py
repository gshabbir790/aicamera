
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class AssetManager:
    def __init__(self, root: str | Path, metadata_file: str | Path | None = None):
        self.root = Path(root)
        self.metadata_file = (
            Path(metadata_file) if metadata_file else self.root / "metadata.json"
        )
        self._items: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Missing metadata: {self.metadata_file}")
        data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("metadata.json must contain a JSON array")

        items: dict[str, dict[str, Any]] = {}
        for item in data:
            self._validate_item(item)
            items[item["id"]] = item
        self._items = items

    @staticmethod
    def _validate_item(item: dict[str, Any]) -> None:
        required = {
            "id", "category", "filename", "shoulder_left_anchor",
            "shoulder_right_anchor", "neck_center_anchor",
            "torso_bottom_anchor", "scale_factor",
        }
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Suit metadata missing keys: {sorted(missing)}")
        if item["category"] not in {"male", "female"}:
            raise ValueError(f"Unsupported suit category: {item['category']}")

        for key in (
            "shoulder_left_anchor",
            "shoulder_right_anchor",
            "neck_center_anchor",
            "torso_bottom_anchor",
        ):
            value = item[key]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"{key} must be [x, y]")

    def get(self, suit_id: str) -> dict[str, Any]:
        try:
            return self._items[suit_id]
        except KeyError as exc:
            raise KeyError(f"Unknown suit_id: {suit_id}") from exc

    def asset_path(self, item: dict[str, Any]) -> Path:
        category = item["category"]
        path = (self.root / category / item["filename"]).resolve()
        root = self.root.resolve()
        if root not in path.parents:
            raise ValueError("Invalid suit asset path")
        return path

    def load_rgba(self, suit_id: str) -> np.ndarray:
        item = self.get(suit_id)
        path = self.asset_path(item)
        if not path.exists():
            raise FileNotFoundError(f"Suit asset not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to decode suit asset: {path}")
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Suit must be RGBA PNG: {path}")
        return image

    def thumbnail_data_url(self, suit_id: str, max_size: int = 180) -> str | None:
        try:
            image = self.load_rgba(suit_id)
        except (OSError, ValueError, KeyError):
            return None

        h, w = image.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 1:
            image = cv2.resize(
                image,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None
        return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")

    def catalog(self) -> list[dict[str, Any]]:
        result = []
        for item in self._items.values():
            result.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "filename": item["filename"],
                    "thumbnail": self.thumbnail_data_url(item["id"]),
                }
            )
        return sorted(result, key=lambda x: (x["category"], x["id"]))
