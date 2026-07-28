document.addEventListener('DOMContentLoaded', () => {
  const pushBtn = document.getElementById('push-btn');
  if (pushBtn) {
    pushBtn.addEventListener('click', async () => {
      const status = document.getElementById('push-status');
      status.textContent = 'Pushing…';
      try {
        const res = await fetch('/push', { method: 'POST' });
        const j = await res.json();
        status.textContent = `Pushed ${j.pushed} entries.`;
      } catch (e) {
        status.textContent = 'Push failed: ' + e;
      }
    });
  }

  const pdfName = window.PDF_NAME;
  if (!pdfName) return;

  const inputsOf = (el) =>
    Object.fromEntries(
      Array.from(el.querySelectorAll('input')).map((i) => [i.name, i.value])
    );

  document.querySelectorAll('article.entry').forEach((el) => {
    const id = el.dataset.id;

    el.querySelector('.accept').addEventListener('click', async () => {
      const res = await fetch(
        `/entry/${encodeURIComponent(pdfName)}/${encodeURIComponent(id)}/accept`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(inputsOf(el)),
        }
      );
      const j = await res.json();
      if (j.ok) el.querySelector('.status-val').textContent = j.entry.review_status;
    });

    el.querySelector('.edit').addEventListener('click', () => openCropper(el, id));
  });

  const dlg = document.getElementById('crop-dialog');
  const img = document.getElementById('crop-target');
  let cropper = null;
  let currentEl = null;
  let currentId = null;

  function openCropper(el, id) {
    currentEl = el;
    currentId = id;
    const page = String(el.dataset.page).padStart(4, '0');
    img.src = `/processed/${encodeURIComponent(pdfName)}/pages/page_${page}.png`;
    dlg.showModal();
    img.onload = () => {
      if (cropper) cropper.destroy();
      cropper = new Cropper(img, { viewMode: 1, autoCrop: true });
    };
  }

  document.getElementById('crop-cancel').addEventListener('click', () => {
    if (cropper) { cropper.destroy(); cropper = null; }
    dlg.close();
  });

  document.getElementById('crop-save').addEventListener('click', async () => {
    if (!cropper) return;
    const d = cropper.getData({ rounded: true });
    const body = {
      x: d.x, y: d.y, w: d.width, h: d.height,
      ...inputsOf(currentEl),
    };
    const res = await fetch(
      `/entry/${encodeURIComponent(pdfName)}/${encodeURIComponent(currentId)}/edit`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }
    );
    const j = await res.json();
    if (j.ok) {
      currentEl.querySelector('.status-val').textContent = j.entry.review_status;
      const cropImg = currentEl.querySelector('.crop img');
      cropImg.src = cropImg.src.split('?')[0] + '?t=' + Date.now();
    }
    if (cropper) { cropper.destroy(); cropper = null; }
    dlg.close();
  });
});
