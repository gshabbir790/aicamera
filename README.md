
# Admission Photo AI Backend

Production-oriented FastAPI additive AI layer for the existing
`admission-photo-camera.html`.

## Important

The supplied frontend is the source of truth. Do not replace its UI.

The backend requires MediaPipe Tasks API model files:

- `models/face_landmarker.task`
- `models/pose_landmarker_full.task`

Obtain these model files from the official MediaPipe model distribution and
place them in `models/`.

Replace the demo transparent suit PNGs in:

- `assets/suits/male/suit_01.png`
- `assets/suits/female/suit_01.png`

with professionally designed RGBA suit templates. The demo assets exist only
to make the catalogue endpoint immediately testable.

## Local

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

## Docker

```bash
docker-compose up --build
```

## API

- `GET /api/v1/health`
- `GET /api/v1/suits`
- `POST /api/v1/preview-suit`
- `POST /api/v1/process-photo`
- `POST /api/v1/process-photo/base64`

## Frontend

1. Keep the complete existing `admission-photo-camera.html`.
2. Insert `frontend-controls-snippet.html` inside its existing `#editModal`.
3. Merge `admission-photo-ai-integration.js` into the existing `<script>`.
4. Set:
   `window.ADMISSION_PHOTO_AI_API = 'http://localhost:8000/api/v1';`
   before the integration code.
5. Preserve the existing edit handlers; add the AI hooks rather than replacing
   the brightness/contrast/sharpen implementation.

## Data contract

Every successful AI modification follows:

```js
photos[id].dataUrl = processed_image_base64;
delete resizedMap[id];
renderGrid();
refreshActionState();
scheduleAutoSave();
```

The original is retained as `photos[id].rawOriginal`.

## Security

Do not log image bytes, Base64 payloads, student names, or other student
information. Configure a restrictive `FRONTEND_ORIGIN` in production.
