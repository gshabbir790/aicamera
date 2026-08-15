
/*
 * AI integration for admission-photo-camera.html
 *
 * IMPORTANT:
 * 1) This is an additive patch. Do not replace the existing application.
 * 2) Insert the HTML controls from the companion snippet inside #editModal,
 *    immediately after #editPreviewCanvas.
 * 3) Replace only the existing openEditModal/updateEditPreview/editApplyBtn
 *    section with the guarded hooks below, or merge these functions carefully.
 */

const AI_API_BASE = window.ADMISSION_PHOTO_AI_API || 'http://localhost:8000/api/v1';
const AI_TIMEOUT_MS = 45000;

let aiSuitCatalog = [];
let aiPreviewTimer = null;
let aiOriginalById = new Map();

function aiToast(message) {
  statusLine.textContent = message;
  statusLine.style.color = 'var(--maroon)';
  window.setTimeout(() => { statusLine.style.color = ''; }, 4500);
}

async function aiFetch(url, options = {}, timeoutMs = AI_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {...options, signal: controller.signal});
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('AI_TIMEOUT');
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

async function loadAISuitCatalog() {
  try {
    const response = await aiFetch(`${AI_API_BASE}/suits`, {}, 10000);
    if (!response.ok) throw new Error(`HTTP_${response.status}`);
    aiSuitCatalog = await response.json();

    const select = $('aiSuitSelector');
    if (!select) return;

    select.innerHTML = '<option value="">سوٹ منتخب نہ کریں</option>';
    aiSuitCatalog.forEach(item => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${item.category === 'female' ? 'خواتین' : 'مردانہ'} — ${item.id}`;
      select.appendChild(option);
    });
  } catch (err) {
    aiSuitCatalog = [];
    const select = $('aiSuitSelector');
    if (select) {
      select.innerHTML = '<option value="">AI سرور دستیاب نہیں</option>';
    }
    aiToast('AI سرور دستیاب نہیں؛ بنیادی ایڈیٹنگ معمول کے مطابق جاری رہے گی۔');
  }
}

function aiEnsureOriginal(p) {
  if (!p) return;
  if (!p.rawOriginal) p.rawOriginal = p.dataUrl;
  aiOriginalById.set(p.id, p.rawOriginal);
}

async function aiProcessCurrentPhoto({preview = false} = {}) {
  const p = photos.find(x => x.id === editTargetId);
  if (!p) return;

  aiEnsureOriginal(p);

  const sourceDataUrl = p.rawOriginal;
  const blob = await (await fetch(sourceDataUrl)).blob();

  const form = new FormData();
  form.append('image', blob, 'student.jpg');
  form.append('suit_id', $('aiSuitSelector')?.value || '');
  form.append('skin_smooth_level',
    String(Number($('aiSkinSmooth')?.value || 0) / 100));
  form.append('brighten_level',
    String(Number($('aiSkinBright')?.value || 0) / 100));

  try {
    const endpoint = preview
      ? `${AI_API_BASE}/preview-suit`
      : `${AI_API_BASE}/process-photo`;

    const response = await aiFetch(endpoint, {method:'POST', body:form}, preview ? 15000 : AI_TIMEOUT_MS);
    if (!response.ok) throw new Error(`HTTP_${response.status}`);

    const result = await response.json();
    if (!result.processed_image) throw new Error('EMPTY_AI_RESULT');

    if (preview) {
      const previewImg = await loadImage(result.processed_image);
      editPreviewCanvas.width = previewImg.naturalWidth || previewImg.width;
      editPreviewCanvas.height = previewImg.naturalHeight || previewImg.height;
      editPreviewCanvas.getContext('2d').drawImage(previewImg, 0, 0);
      return;
    }

    p.dataUrl = result.processed_image;
    delete resizedMap[p.id];

    renderGrid();
    refreshActionState();
    scheduleAutoSave();

    editSourceImg = await loadImage(p.dataUrl);
    updateEditPreview();

    statusLine.textContent = result.metadata?.error_code
      ? `AI مکمل نہیں ہوئی: ${result.metadata.error_code}`
      : 'AI سوٹ اور فیس ٹچ اپ کامیابی سے لاگو ہو گیا۔';
  } catch (err) {
    if (err.message === 'AI_TIMEOUT') {
      aiToast('AI سرور کا جواب مقررہ وقت میں نہیں آیا۔ اصل ایپ محفوظ ہے۔');
    } else {
      aiToast('AI پراسیسنگ دستیاب نہیں؛ براہِ کرم سرور چیک کریں۔');
    }
  }
}

function aiRevertCurrent() {
  const p = photos.find(x => x.id === editTargetId);
  if (!p) return;

  aiEnsureOriginal(p);
  p.dataUrl = p.rawOriginal;
  delete resizedMap[p.id];

  renderGrid();
  refreshActionState();
  scheduleAutoSave();

  editSourceImg = null;
  loadImage(p.dataUrl).then(img => {
    editSourceImg = img;
    updateEditPreview();
  });
}

function aiSchedulePreview() {
  if (!$('aiTouchupToggle')?.checked) return;
  if (aiPreviewTimer) clearTimeout(aiPreviewTimer);
  aiPreviewTimer = setTimeout(() => {
    aiProcessCurrentPhoto({preview:true}).catch(() => {});
  }, 500);
}

async function aiInitEditModal(id) {
  const p = photos.find(x => x.id === id);
  if (!p) return;
  aiEnsureOriginal(p);

  const toggle = $('aiTouchupToggle');
  if (toggle) toggle.checked = false;
  if ($('aiSkinSmooth')) $('aiSkinSmooth').value = 30;
  if ($('aiSkinBright')) $('aiSkinBright').value = 10;
  if ($('aiSkinSmoothVal')) $('aiSkinSmoothVal').textContent = '30%';
  if ($('aiSkinBrightVal')) $('aiSkinBrightVal').textContent = '10%';
  if ($('aiSuitSelector')) $('aiSuitSelector').value = '';
}

/*
 * Merge these handlers into the existing Edit Modal code.
 * The existing brightness/contrast/sharpen controls remain untouched.
 */
$('aiTouchupToggle')?.addEventListener('change', aiSchedulePreview);
$('aiSkinSmooth')?.addEventListener('input', e => {
  $('aiSkinSmoothVal').textContent = `${e.target.value}%`;
  aiSchedulePreview();
});
$('aiSkinBright')?.addEventListener('input', e => {
  $('aiSkinBrightVal').textContent = `${e.target.value}%`;
  aiSchedulePreview();
});
$('aiSuitSelector')?.addEventListener('change', aiSchedulePreview);
$('aiApplyBtn')?.addEventListener('click', () => aiProcessCurrentPhoto({preview:false}));
$('aiRevertBtn')?.addEventListener('click', aiRevertCurrent);

loadAISuitCatalog();
