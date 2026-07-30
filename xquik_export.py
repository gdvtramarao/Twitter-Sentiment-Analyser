"""Read text rows from Xquik extraction exports."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from typing import Any

MAX_EXPORT_BYTES = 10 * 1024 * 1024
MAX_EXPORT_ROWS = 5_000
TEXT_FIELDS = ("tweettext", "tweet", "fulltext", "text", "content", "body")
WRAPPER_FIELDS = ("data", "results", "tweets", "items")


class XquikExportError(ValueError):
    """Raised when an Xquik export cannot be read safely."""


def _iter_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        for key in WRAPPER_FIELDS:
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                yield from _iter_records(nested)
                return
        yield value
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_records(item)


def _read_json_or_jsonl(raw: str) -> Iterable[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, RecursionError):
        for line_number, line in enumerate(raw.splitlines(), start=1):
            candidate = line.strip()
            if not candidate:
                continue
            try:
                yield from _iter_records(json.loads(candidate))
            except (json.JSONDecodeError, RecursionError) as error:
                raise XquikExportError(
                    f"Invalid JSON on line {line_number}. Export JSON, JSONL, or CSV again."
                ) from error
        return

    yield from _iter_records(parsed)


def _read_csv(raw: str) -> Iterable[dict[str, Any]]:
    try:
        yield from csv.DictReader(io.StringIO(raw))
    except csv.Error as error:
        raise XquikExportError("Invalid CSV. Export the file again.") from error


def _normalize_field_name(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _first_text(row: dict[str, Any]) -> str:
    normalized = {
        _normalize_field_name(field): value
        for field, value in row.items()
        if field is not None
    }
    for field in TEXT_FIELDS:
        value = normalized.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_xquik_texts(
    payload: bytes,
    *,
    max_bytes: int = MAX_EXPORT_BYTES,
    max_rows: int = MAX_EXPORT_ROWS,
) -> list[str]:
    """Return text from an Xquik JSON, JSONL, or CSV export."""
    if len(payload) > max_bytes:
        raise XquikExportError(
            f"File exceeds the {max_bytes // (1024 * 1024)} MB limit."
        )

    try:
        raw = payload.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise XquikExportError("File must use UTF-8 encoding.") from error
    if not raw:
        return []

    records = _read_json_or_jsonl(raw) if raw[:1] in "[{" else _read_csv(raw)
    texts: list[str] = []
    try:
        for row_number, record in enumerate(records, start=1):
            if row_number > max_rows:
                raise XquikExportError(f"File exceeds the {max_rows:,}-row limit.")
            text = _first_text(record)
            if not text:
                continue
            texts.append(text)
    except RecursionError as error:
        raise XquikExportError("Export nesting is too deep.") from error
    return texts
