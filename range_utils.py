from __future__ import annotations

import re


_PART_RE = re.compile(r"^(\d+)(?:\s*[-–—]\s*(\d+))?$")


def parse_page_range(spec: str | None, total_pages: int) -> list[int]:
    """Parse `all`, `1-5`, or `1-3, 8, 12-15` into sorted one-based pages."""
    if total_pages < 1:
        return []
    value = (spec or "all").strip().lower()
    if value in {"", "all", "全部", "*"}:
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for raw_part in re.split(r"[,，]", value):
        part = raw_part.strip()
        match = _PART_RE.match(part)
        if not match:
            raise ValueError(f"Invalid page range: {raw_part!r}.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            raise ValueError(f"Page range must run forward: {part}.")
        if start < 1 or end > total_pages:
            raise ValueError(f"Page range {part} is outside 1-{total_pages}.")
        pages.update(range(start, end + 1))
    return sorted(pages)
