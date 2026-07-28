from __future__ import annotations
import os

from dotenv import load_dotenv

from .logging_setup import get_logger

log = get_logger("push")


def push_all(processed_root: str = "processed") -> int:
    """Iterate accepted/edited, not-yet-pushed entries and push them to Supabase.

    Currently a stub: gated on SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY in .env.
    When both are present, implement:
      1. upload crops/<entry_id>.png to Supabase Storage (bucket: signatures)
      2. insert row into `signatures` table with storage URL + metadata
      3. mark entry `pushed=True` in its JSON sidecar
      4. write result to logs/push.log
    """
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        log.warning("Supabase credentials not set — push disabled. "
                    "Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env "
                    "when ready.")
        return 0
    log.info("Supabase credentials found, but push logic is not yet implemented.")
    return 0
