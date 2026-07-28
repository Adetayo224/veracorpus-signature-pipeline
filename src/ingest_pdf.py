from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import fitz  # PyMuPDF

from .logging_setup import get_logger

log = get_logger("pipeline")


def _render_page(args):
    pdf_path, page_index, out_dir, dpi = args
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = Path(out_dir) / f"page_{page_index + 1:04d}.png"
        pix.save(str(out_path))
        return page_index, str(out_path)
    finally:
        doc.close()


def render_pdf(pdf_path: str, out_dir: str, dpi: int = 300,
               workers: int | None = None) -> list[str]:
    pdf_path = str(Path(pdf_path).resolve())
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()
    log.info(f"Rendering {n_pages} pages from {Path(pdf_path).name} at {dpi} DPI")
    tasks = [(pdf_path, i, str(out_dir_p), dpi) for i in range(n_pages)]
    results: list[tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_render_page, tasks):
            results.append(res)
    results.sort(key=lambda x: x[0])
    return [p for _, p in results]
