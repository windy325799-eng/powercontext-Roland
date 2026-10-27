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

"""Capture the visible events of one selected Pi session branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_EVENT_DEFINITION_VERSION = "1"
_CHECKPOINT_SCHEMA = "powercontext.pi-session-checkpoint.v1"


class PiSessionInput(BaseModel):
    """One complete Pi v3 session archive and the branch leaf to select."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible event on the selected ancestor path."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured original event, immutable under its session and event identity."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Return the visible events of the selected branch from root to leaf.

    Invalid archives fail atomically before any event is produced. Nodes without a
    visible body keep their place in the parent graph but are omitted from the result.
    """

    records = _read_records(value.content)
    if not records:
        raise ValueError("Pi session must start with a session header record")
    header = records[0]
    if header.get("type") != "session":
        raise ValueError("Pi session must start with a type=session record")
    if type(header.get("version")) is not int or header.get("version") != 3:
        raise ValueError("Pi session header must declare version=3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Pi session header must declare a non-empty id")

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("Pi session records must declare a non-empty id")
        if event_id in by_id:
            raise ValueError("Pi session record ids must be unique")
        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("Pi session parentId must be a string or null")
        if isinstance(parent_id, str) and parent_id:
            if parent_id not in by_id:
                raise ValueError("Pi session parentId must reference an earlier record")
        by_id[event_id] = record
        order.append(event_id)

    leaf_id = value.leaf_id
    if leaf_id is None:
        if not order:
            return ()
        leaf_id = order[-1]
    if not isinstance(leaf_id, str) or leaf_id not in by_id:
        raise ValueError("Pi session leaf_id does not identify a record")

    chain: list[dict[str, Any]] = []
    current: str | None = leaf_id
    while current is not None:
        record = by_id[current]
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) and parent else None
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        event = _visible_event(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable identity of one event within its session."""

    encoded = f"pi:{_quote(session_id)}:{_quote(event_id)}"
    if len(encoded) <= MAX_SOURCE_ID_LENGTH:
        return encoded
    digest = hashlib.sha256(f"{session_id}\0{event_id}".encode("utf-8")).hexdigest()
    return f"pi:sha256:{digest}"


class PiEventSourceAdapter:
    """Resolve and read one captured Pi event without touching the session file."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            definition_version=PI_EVENT_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose the complete event text with its session and graph metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class PiSessionConnector:
    """Import the unvisited events of one selected branch and checkpoint progress."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_EVENT_DEFINITION_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=content, leaf_id=self._leaf_id))
        imported = _imported_records(session.checkpoint)

        pending: dict[str, str] = {}
        for event in events:
            digest = _record_digest(event.record)
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError("Pi session rewrote an event that was already imported")
                continue
            pending[event.event_id] = digest

        for event in events:
            if event.event_id not in pending:
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)

        merged = {**imported, **pending}
        checkpoint: dict[str, JsonValue] = {"schema": _CHECKPOINT_SCHEMA, "events": merged}
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_DEFINITION_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


def _read_records(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be strict UTF-8") from error
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session content must be one JSON object per line") from error
        if not isinstance(record, dict):
            raise ValueError("Pi session content must be one JSON object per line")
        records.append(record)
    return records


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    event_type = record.get("type")
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message events must contain a message object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Pi message events must declare a role")
        body = _message_body(role, message)
        if body is None:
            return None
        event_role = role
    elif event_type in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if not isinstance(summary, str):
            raise ValueError(f"Pi {event_type} events must contain a summary")
        if not summary.strip():
            return None
        body = summary
        event_role = event_type
    elif event_type == "custom_message":
        body = _text_content(record.get("content"))
        if not body.strip():
            return None
        event_role = "custom"
    else:
        return None

    parent_id = record.get("parentId")
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=parent_id if isinstance(parent_id, str) and parent_id else None,
        event_type=event_type,
        role=event_role,
        content=f"{event_role}: {body}",
        record=record,
    )


def _message_body(role: str, message: dict[str, Any]) -> str | None:
    if role == "toolResult":
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
            raise ValueError("Pi tool results must declare toolName and toolCallId")
        return f"[tool_result {tool_name} {tool_call_id}] {_text_content(message.get('content'))}"
    if role == "bashExecution":
        command = message.get("command")
        output = message.get("output")
        if not isinstance(command, str):
            raise ValueError("Pi bash execution events must declare a command")
        if output is not None and not isinstance(output, str):
            raise ValueError("Pi bash execution output must be a string")
        return f"$ {command}\n{output or ''}"
    parts = _message_parts(message.get("content"))
    if not parts:
        return None
    return "\n".join(parts)


def _message_parts(content: object) -> list[str]:
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError("Pi tool calls must declare name and id")
            arguments = block.get("arguments")
            parts.append(f"[tool_call {name} {call_id}] {_compact_json(arguments)}")
    return parts


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    return "\n".join(texts)


def _compact_json(value: object) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _imported_records(checkpoint: JsonValue | None) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("Pi session checkpoint must be a JSON object")
    events = checkpoint.get("events")
    if events is None:
        return {}
    if not isinstance(events, dict) or not all(isinstance(value, str) for value in events.values()):
        raise ValueError("Pi session checkpoint events must map ids to digests")
    return {str(event_id): digest for event_id, digest in events.items()}


def _quote(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Pi event identity components must be non-empty strings")
    return quote(value, safe="")


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
