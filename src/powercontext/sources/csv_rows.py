# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One durable Source per CSV record, addressed by document and primary key."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

CSV_ROW_SOURCE_NAME = "csv-row"
_CSV_IDENTITY_PREFIX = "csv"


class CSVDocumentInput(BaseModel):
    """Complete offline CSV bytes to split into one Source per record."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One validated CSV record in original header order."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured record whose identity is its document and primary-key value."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate a CSV document before returning any record.

    The first record is the header and every later record maps exactly one Source.
    All validation happens before the tuple is returned, so a malformed document
    never yields a prefix of its rows.
    """

    document_id = _validate_document_id(value.document_id)
    primary_key = _validate_primary_key(value.primary_key)
    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV content must be strict UTF-8") from error
    if not text:
        raise ValueError("CSV document must contain a header record")

    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"CSV document is malformed: {error}") from error
    if not records:
        raise ValueError("CSV document must contain a header record")

    header = records[0]
    columns = tuple(header)
    if not columns:
        raise ValueError("CSV header must not be empty")
    for column in columns:
        if not column.strip():
            raise ValueError("CSV header fields must be non-empty and not whitespace")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header fields must be unique")
    try:
        key_index = columns.index(primary_key)
    except ValueError as error:
        raise ValueError(f"primary key {primary_key!r} does not match a CSV header field") from error

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            continue
        if len(record) != len(columns):
            raise ValueError("CSV record width does not match the header")
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary-key values must be non-empty and not whitespace")
        if primary_key_value in seen_keys:
            raise ValueError("CSV primary-key values must be unique in one document")
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable, delimiter-safe identity of one CSV record."""

    document_id = _validate_document_id(document_id)
    if not isinstance(primary_key_value, str) or not primary_key_value:
        raise ValueError("primary-key value must be a non-empty string")
    source_id = f"{_CSV_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError("CSV row identity must not exceed the Source identity length limit")
    return source_id


class CSVRowSourceAdapter:
    """Resolve validated rows and read their exact captured fields back."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        return CSVRowSource(
            name=csv_row_source_id(value.document_id, value.primary_key_value),
            materialization=SourceMaterialization.CAPTURED,
            source_id=csv_row_source_id(value.document_id, value.primary_key_value),
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one row as ordered JSON text with its exact record metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        if len(source.columns) != len(source.values):
            raise ValueError("CSV row columns and values must have the same width")
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.name,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    projections=(CSVRowTextEvidenceProjection(),),
)


def _validate_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")
    return document_id


def _validate_primary_key(primary_key: object) -> str:
    if not isinstance(primary_key, str) or not primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")
    return primary_key


__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
