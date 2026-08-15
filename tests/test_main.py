
from __future__ import annotations


import cv2
import numpy as np
from dataclasses import replace
from fastapi.testclient import TestClient

import main


def _jpeg_bytes():
    image = np.full((120, 100, 3), 200, dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


class FakeAssets:
    def catalog(self):
        return []


class FakePipeline:
    def process(self, image, suit_id, smooth, brighten):
        class Result:
            pass

        result = Result()
        result.face_detected = True
        result.pose_detected = True
        result.landmark_count = 500
        result.landmark_confidence = 0.9
        result.suit_applied = bool(suit_id)
        result.skin_retouch_applied = smooth > 0 or brighten > 0
        result.error_code = None
        result.image = image
        return result


class NoFacePipeline(FakePipeline):
    def process(self, image, suit_id, smooth, brighten):
        result = super().process(image, suit_id, smooth, brighten)
        result.face_detected = False
        result.pose_detected = False
        result.landmark_count = 0
        result.landmark_confidence = 0.0
        result.error_code = "NO_FACE"
        return result


class LowConfidencePipeline(FakePipeline):
    def process(self, image, suit_id, smooth, brighten):
        result = super().process(image, suit_id, smooth, brighten)
        result.landmark_confidence = 0.4
        result.suit_applied = False
        result.error_code = "LANDMARK_CONFIDENCE_LOW"
        return result


def test_invalid_image():
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = FakePipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert response.status_code == 400


def test_oversized_upload(monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, max_upload_bytes=4))
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = FakePipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("x.jpg", b"12345", "image/jpeg")},
        )
        assert response.status_code == 413


def test_health():
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_base64_validation():
    with TestClient(main.app) as client:
        response = client.post(
            "/api/v1/process-photo/base64",
            json={"image": "not-a-data-url"},
        )
        assert response.status_code == 400


def test_success_pipeline(monkeypatch):
    monkeypatch.setattr(main, "settings", replace(main.settings, max_upload_bytes=15 * 1024 * 1024))
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = FakePipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
            data={
                "suit_id": "suit_male_01",
                "skin_smooth_level": "0.3",
                "brighten_level": "0.1",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["metadata"]["suit_applied"] is True
        assert payload["processed_image"].startswith("data:image/jpeg;base64,")


def test_no_face_fallback():
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = NoFacePipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is False
        assert payload["metadata"]["error_code"] == "NO_FACE"


def test_low_confidence_diagnostic():
    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = LowConfidencePipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["metadata"]["landmark_confidence"] == 0.4
        assert payload["metadata"]["suit_applied"] is False


def test_invalid_suit_asset_diagnostic():
    class InvalidSuitPipeline(FakePipeline):
        def process(self, image, suit_id, smooth, brighten):
            result = super().process(image, suit_id, smooth, brighten)
            result.suit_applied = False
            result.error_code = "INVALID_SUIT_ASSET"
            return result

    with TestClient(main.app) as client:
        main.app.state.assets = FakeAssets()
        main.app.state.pipeline = InvalidSuitPipeline()
        response = client.post(
            "/api/v1/process-photo",
            files={"image": ("x.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"suit_id": "missing"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert response.json()["metadata"]["error_code"] == "INVALID_SUIT_ASSET"
