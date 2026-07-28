from __future__ import annotations
import argparse
import time
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.logging_setup import get_logger
from src.ingest_pdf import render_pdf
from src.detect_regions import detect_page_entries
from src.crop_signature import (
    compute_global_v_xs,
    crop_signature_from_band,
    crop_signature_to_cell,
    detect_page_rules,
    repair_outlier_crops,
)
from src.store_locally import pdf_workdir, write_entry, write_manifest

log = get_logger("pipeline")


def _process_page(args: tuple) -> tuple[int, list[dict]]:
    """Worker: given a rendered page + its PDF metadata, return the list of
    entry dicts ready to be written. Runs in a subprocess."""
    (pdf_path, page_index, page_image_path, workdir_str, dpi,
     v_xs_global) = args
    workdir = Path(workdir_str)

    entries = detect_page_entries(pdf_path, page_index, dpi=dpi)
    h_ys, v_xs = detect_page_rules(page_image_path)
    # Fall back to the pooled cross-page template when this page's own
    # vertical-rule detection came up short.
    if len(v_xs) < 3 and v_xs_global:
        v_xs = v_xs_global
    median_row_h = None
    if len(h_ys) >= 3:
        diffs = [h_ys[i + 1] - h_ys[i] for i in range(len(h_ys) - 1)]
        diffs.sort()
        median_row_h = int(diffs[len(diffs) // 2])
    out: list[dict] = []
    for i, e in enumerate(entries):
        entry_id = f"p{page_index + 1:04d}_e{i:02d}"
        crop_path = workdir / "crops" / f"{entry_id}.png"
        sig_bbox = crop_signature_to_cell(
            page_image_path, e.signature_region, h_ys, v_xs, str(crop_path),
            median_row_h=median_row_h,
        )
        if sig_bbox is None:
            sig_bbox = crop_signature_from_band(
                page_image_path, e.signature_region, str(crop_path)
            )
        if sig_bbox is None:
            continue
        crop_method = sig_bbox.pop("method", "ink_projection")
        out.append({
            "id": entry_id,
            "page_number": page_index + 1,
            "artist_name_claimed": e.artist_name_claimed,
            "dates": e.dates,
            "title": e.title,
            "raw_ocr_text": e.raw_text,
            "text_region_bbox": e.text_region.to_dict(),
            "signature_search_bbox": e.signature_region.to_dict(),
            "signature_bbox": sig_bbox,
            "crop_method": crop_method,
        })
    return page_index, out


def process_pdf(pdf_path: Path, dpi: int = 150,
                workers: int | None = None) -> tuple[int, float]:
    t0 = time.time()
    workdir = pdf_workdir(str(pdf_path))
    pages_dir = workdir / "pages"
    log.info(f"processing {pdf_path.name} -> {workdir}")

    page_paths = render_pdf(str(pdf_path), str(pages_dir),
                            dpi=dpi, workers=workers)
    t_render = time.time() - t0
    log.info(f"rendered {len(page_paths)} pages in {t_render:.1f}s")

    # Build a cross-page vertical-rule template so pages whose own detection
    # is weak still get correct left/right cell bounds.
    v_xs_global = compute_global_v_xs(page_paths)
    log.info(f"global v-rule template: {v_xs_global}")

    tasks = [
        (str(pdf_path), i, page_paths[i], str(workdir), dpi, v_xs_global)
        for i in range(len(page_paths))
    ]

    entry_count = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_page, t) for t in tasks]
        for fut in as_completed(futures):
            _, entries = fut.result()
            for entry in entries:
                entry["source_pdf"] = pdf_path.name
                write_entry(workdir, entry)
                entry_count += 1

    # Outlier-repair pass: any crop whose w or h is far below the page's
    # median gets re-cropped using median cell dimensions centred on its
    # search band. This is the safety net for cells whose rules escaped
    # detection entirely.
    n_repaired = repair_outlier_crops(workdir, v_xs_global)
    if n_repaired:
        log.info(f"outlier repair: fixed {n_repaired} crops")

    write_manifest(workdir, pdf_path.name)
    elapsed = time.time() - t0
    log.info(f"done: {pdf_path.name} — {entry_count} entries across "
             f"{len(page_paths)} pages in {elapsed:.1f}s "
             f"(render {t_render:.1f}s, detect+crop {elapsed - t_render:.1f}s)")
    return entry_count, elapsed


def run_review() -> None:
    import uvicorn
    url = "http://127.0.0.1:8000"
    log.info(f"launching review app at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run("review_app.server:app", host="127.0.0.1",
                port=8000, reload=False)


def run_push() -> None:
    from src.push_to_supabase import push_all
    n = push_all()
    log.info(f"push finished: {n} entries pushed")


def main() -> None:
    ap = argparse.ArgumentParser(description="VeraCorpus signature pipeline")
    ap.add_argument("--input", help="path to a PDF or folder of PDFs")
    ap.add_argument("--dpi", type=int, default=150,
                    help="render DPI (default 150 — source scans are ~72 DPI native)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: os cpu count)")
    ap.add_argument("--no-review", action="store_true",
                    help="do not launch the review UI after processing")
    ap.add_argument("--review", action="store_true",
                    help="only launch the review UI (skip processing)")
    ap.add_argument("--push", action="store_true",
                    help="push accepted/edited entries to Supabase")
    ap.add_argument("--flag", action="store_true",
                    help="tag every entry with crop_quality based on per-PDF "
                         "median bbox size; rewrites entries + packets")
    ap.add_argument("--recrop", action="store_true",
                    help="retry cropping for entries flagged as too_small / "
                         "too_narrow / etc, using a widened search band and a "
                         "more forgiving projection; requires --flag first")
    args = ap.parse_args()

    if args.push:
        run_push()
        return
    if args.review:
        run_review()
        return
    if args.flag:
        from src.flag_outliers import flag_all
        for r in flag_all():
            print(f"{r['pdf']}: {r['total']} entries, "
                  f"{r['flagged']} flagged — {r['by_class']}")
        return
    if args.recrop:
        from src.recrop_flagged import recrop_all
        for r in recrop_all():
            print(f"{r['pdf']}: retried {r['retried']}, "
                  f"improved {r['improved']}, kept {r['kept_original']}, "
                  f"still_bad {r['still_bad']}")
        return
    if not args.input:
        ap.error("--input is required unless --review or --push is set")

    inp = Path(args.input)
    if inp.suffix.lower() == ".pdf":
        pdfs = [inp]
    else:
        pdfs = sorted(inp.glob("*.pdf"))
    if not pdfs:
        ap.error(f"no PDFs found at {inp}")

    total_entries = 0
    t0 = time.time()
    for pdf in pdfs:
        n, _ = process_pdf(pdf, dpi=args.dpi, workers=args.workers)
        total_entries += n
    log.info(f"all done: {len(pdfs)} PDF(s), {total_entries} entries, "
             f"{time.time() - t0:.1f}s total")

    if not args.no_review:
        run_review()


if __name__ == "__main__":
    main()
