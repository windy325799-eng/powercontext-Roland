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

"""Page-level PDF Sources with exact, printable evidence citations."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import quote

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization, SourceRef

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_DEFINITION_VERSION = "1"
_PDF_IDENTITY_PREFIX = "pdf"

_TOKEN = re.compile(r"[a-z]+|\d+")

# Function words add no evidence to a page lookup.
_STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below between both
    but by can could did do does doing down during each either few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just may me might more most much must my myself no nor not
    now of off on once only or other ought our ours ourselves out over own same shall she should so some such than that
    the their theirs them themselves then there these they this those through to too under until up very was we
    were what when where which while who whom why will with within would you your yours yourself yourselves
    """.split()
)
# Question scaffolding and answer instructions are not document evidence.
_SCAFFOLD = frozenset(
    """
    according answer assume based business calculation clarify compare comparison data define definition detail details
    directly explain explicit explicitly fiscal form give important improving instruction materially metric metrics
    noted ongoing outlined perspective please provide primarily quantity question reasonably relevant report reported
    reporting reports respond response shown state stated useful using way
    """.split()
)
_STEM_STOP = _STOPWORDS | _SCAFFOLD

# Generic finance lexicon: members of one group are interchangeable descriptions of the
# same report concept, including the table titles that financial statements use.
_THESAURUS: tuple[tuple[str, ...], ...] = (
    ("revenue", "net revenue", "net revenues", "sales", "net sales", "total revenue", "total revenues", "turnover"),
    ("operating income", "income from operations", "operating profit", "operating profits", "operating earnings"),
    ("gross profit", "gross profits", "gross margin", "gross margins"),
    ("net income", "net earnings", "net loss", "net income loss", "earnings"),
    (
        "margin",
        "margins",
        "operating margin",
        "gross margin",
        "profit margin",
        "percent of sales",
        "percentage of sales",
    ),
    ("ebitda", "depreciation and amortization", "depreciation", "amortization", "operating income"),
    (
        "capex",
        "capital expenditure",
        "capital expenditures",
        "purchases of property",
        "property plant and equipment",
        "capital spending",
        "capital intensive",
    ),
    (
        "cash flow",
        "cash flows",
        "cashflow",
        "cash provided by operating activities",
        "operating activities",
        "net cash provided by operating activities",
    ),
    ("free cash flow", "free cashflow", "fcf", "cash flow"),
    (
        "liquidity",
        "liquid",
        "current assets",
        "current liabilities",
        "working capital",
        "cash and cash equivalents",
        "quick ratio",
        "current ratio",
    ),
    (
        "quick ratio",
        "acid test",
        "current assets",
        "current liabilities",
        "accounts receivable",
        "short term investments",
    ),
    (
        "tax",
        "taxes",
        "effective tax rate",
        "income tax expense",
        "income taxes",
        "tax rate",
        "provision for income taxes",
        "pretax income",
    ),
    (
        "legal",
        "legal proceedings",
        "litigation",
        "lawsuits",
        "legal actions",
        "claims",
        "contingencies",
        "investigations",
        "regulatory proceedings",
        "legal matters",
        "legal battles",
    ),
    (
        "customer",
        "customers",
        "client",
        "clients",
        "buyers",
        "purchasers",
        "customer concentration",
        "major customer",
        "one customer",
        "concentration of credit risk",
        "commercial airlines",
        "government",
        "government contracts",
        "distributors",
        "retailers",
    ),
    (
        "geography",
        "geographies",
        "geographic",
        "geographic areas",
        "regions",
        "regional",
        "countries",
        "territories",
        "north america",
        "latin america",
        "europe",
        "asia pacific",
        "africa",
        "middle east",
        "amesa",
        "apac",
        "australia",
        "new zealand",
        "china",
    ),
    ("segment", "segments", "reportable segment", "reportable segments", "business segment", "business segments"),
    ("restructuring", "restructuring charges", "restructuring costs", "productivity plan", "impairment charges"),
    (
        "product",
        "products",
        "services",
        "products and services",
        "product categories",
        "service categories",
        "offerings",
        "offering",
    ),
    ("cyclical", "cyclicality", "economic cycles", "cyclical business"),
    ("production", "production rate", "production rates", "production changes", "production levels"),
    ("inventory", "inventories", "cost of sales", "cost of goods sold"),
    ("receivable", "receivables", "accounts receivable", "days sales outstanding"),
    ("dividend", "dividends", "payout", "share repurchase", "repurchases"),
    ("debt", "borrowings", "long term debt", "notes payable", "indebtedness"),
    ("equity", "stockholders equity", "shareholders equity", "stockholders deficit"),
    (
        "growth",
        "increase",
        "increased",
        "decrease",
        "decreased",
        "change",
        "changes",
        "improve",
        "improving",
        "improved",
    ),
    ("sales change", "sales changes", "change in sales", "sales growth", "revenue growth", "segment sales"),
    ("acquisition", "acquisitions", "merger", "mergers", "divestiture", "divestitures", "mergers acquisitions"),
)

_EXPANSION_TERM_WEIGHT = 0.55
_EXPANSION_PHRASE_WEIGHT = 1.3
_PHRASE_WEIGHT = 1.2
_HEADING_WEIGHT = 1.3
_TITLE_WEIGHT = 1.1
_STATEMENT_INJECTION = 0.9
_EXHIBIT_PENALTY = 0.2


class PDFDocumentInput(BaseModel):
    """Complete original PDF bytes to split into physical pages."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical PDF page with its full extractable text."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured page bound to the document revision it was parsed from."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Extract every physical page in order, or fail without partial pages."""

    document_id = _validate_document_id(value.document_id)
    content = value.content
    if not isinstance(content, (bytes, bytearray)) or not content:
        raise ValueError("PDF content must be non-empty bytes")
    raw = bytes(content)
    document_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a user password that was not supplied")
        page_count = len(reader.pages)
        pages = tuple(
            PDFPageInput(
                document_id=document_id,
                document_sha256=document_sha256,
                page_number=page_number,
                text=_extract_page_text(reader, page_number),
            )
            for page_number in range(1, page_count + 1)
        )
    except ValueError:
        raise
    except Exception as error:  # pypdf raises several unrelated parser exceptions.
        raise ValueError(f"PDF document could not be fully parsed: {error}") from error
    return pages


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Return the stable identity of one page in one document revision."""

    document_id = _validate_document_id(document_id)
    if not isinstance(document_sha256, str) or len(document_sha256) != 64 or document_sha256 != document_sha256.lower():
        raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters")
    try:
        int(document_sha256, 16)
    except ValueError as error:
        raise ValueError("document_sha256 must be 64 lowercase hexadecimal characters") from error
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive integer")
    identity = f"{_PDF_IDENTITY_PREFIX}:{quote(document_id, safe='')}:{document_sha256}:{page_number}"
    if len(identity) <= MAX_SOURCE_ID_LENGTH:
        return identity
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    return f"{_PDF_IDENTITY_PREFIX}:{digest}:{document_sha256}:{page_number}"


class PDFPageSourceAdapter:
    """Resolve one parsed page and read its exact text back."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            definition_version=PDF_PAGE_DEFINITION_VERSION,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose the page as document-prefixed text with physical page metadata."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_DEFINITION_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


@dataclass(slots=True)
class _IndexedPage:
    """One page read back from scoped Memory and prepared for lexical ranking."""

    citation: MemoryCitation
    document_id: str
    page_number: int
    text: str
    head_text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    frequency: Counter[str]
    heading_frequency: Counter[str]
    heading_length: int
    title_frequency: Counter[str]
    title_length: int
    statement_kinds: tuple[str, ...]


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct page citations ranked for ``question``.

    Pages are read back through the public scoped Memory API, scored offline, and
    returned as their exact ``MemoryCitation`` so the caller can verify the page text,
    document digest and physical page number instead of trusting a rank.
    """

    if limit <= 0 or not isinstance(question, str) or not question.strip():
        return ()
    pages = await _load_pages(memory)
    if not pages:
        return ()
    ranked = _rank_pages(pages, _analyze_question(question), question)
    selected: list[MemoryCitation] = []
    seen: set[tuple[str, int]] = set()
    for page in ranked:
        location = (page.document_id, page.page_number)
        if location in seen:
            continue
        seen.add(location)
        selected.append(page.citation)
        if len(selected) >= limit:
            break
    return tuple(selected)


async def _load_pages(memory) -> tuple[_IndexedPage, ...]:
    lister = getattr(memory, "list", None)
    if callable(lister):
        listing = await lister()
        if getattr(listing, "memory_ref", None) is None:
            return ()
        pages = tuple(
            page
            for page in (_index_record(record) for record in getattr(listing, "entries", ()))
            if page is not None
        )
        if pages:
            return pages
    return await _search_pages(memory)


async def _search_pages(memory) -> tuple[_IndexedPage, ...]:
    """Fallback candidate pool for a Memory facade without a complete listing."""

    from powercontext.builtin.runtime import GetMemoryEntryRequest

    listing = await memory.search(_search_request("page"))
    pages: list[_IndexedPage] = []
    for hit in getattr(listing, "hits", ()):
        record = await memory.get(
            GetMemoryEntryRequest(
                citation=MemoryCitation(
                    memory_ref=hit.memory_ref,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
            )
        )
        page = _index_record(record)
        if page is not None:
            pages.append(page)
    return tuple(pages)


def _search_request(query: str):
    from powercontext.builtin.runtime import SearchMemoryRequest

    return SearchMemoryRequest(query=query, mode="fts", limit=100)


def _index_record(record) -> _IndexedPage | None:
    entry = getattr(record, "entry", None)
    citation = getattr(record, "citation", None)
    if entry is None or citation is None:
        return None
    lines = entry.text.split("\n", 1)
    document_id = lines[0].strip()
    if not document_id:
        return None
    body = lines[1] if len(lines) > 1 else ""
    page_number, _digest = _location_from_sources(entry.sources)
    if page_number is None:
        return None
    tokens = tuple(stem(token) for token in _tokenize(body))
    head_text = "\n".join(line for line in body.splitlines() if line.strip())[:200]
    title_text = "\n".join([line for line in body.splitlines() if line.strip()][:4])
    heading_tokens = [stem(token) for line in _heading_lines(body) for token in _tokenize(line)]
    title_tokens = [stem(token) for token in _tokenize(title_text)]
    return _IndexedPage(
        citation=citation,
        document_id=document_id,
        page_number=page_number,
        text=body,
        head_text=head_text,
        tokens=tokens,
        token_set=frozenset(tokens),
        frequency=Counter(tokens),
        heading_frequency=Counter(heading_tokens),
        heading_length=max(len(heading_tokens), 1),
        title_frequency=Counter(title_tokens),
        title_length=max(len(title_tokens), 1),
        statement_kinds=_statement_kinds(body),
    )


def _location_from_sources(sources: tuple[SourceRef, ...]) -> tuple[int | None, str]:
    for source in sources:
        if source.source_type != PDF_PAGE_SOURCE_NAME:
            continue
        pieces = source.source_id.rsplit(":", 2)
        if len(pieces) != 3:
            continue
        try:
            return int(pieces[2]), pieces[1]
        except ValueError:
            continue
    return None, ""


def _extract_page_text(reader: PdfReader, page_number: int) -> str:
    page = reader.pages[page_number - 1]
    try:
        text = page.extract_text()
    except Exception as error:
        raise ValueError(f"PDF page {page_number} could not be fully parsed: {error}") from error
    return text or ""


def _validate_document_id(document_id: object) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-blank string")
    return document_id


# ---------------------------------------------------------------------------
# Offline query analysis and ranking
# ---------------------------------------------------------------------------


def _normalize_compounds(text: str) -> str:
    """Spell out finance abbreviations so their concept groups can match."""

    text = re.sub(r"\bm\s*&\s*a\b", "mergers acquisitions", text, flags=re.IGNORECASE)
    text = re.sub(r"\bd\s*&\s*a\b", "depreciation amortization", text, flags=re.IGNORECASE)
    text = re.sub(r"\bp\s*&\s*l\b", "income statement", text, flags=re.IGNORECASE)
    return text


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(_normalize_compounds(text).casefold())
        if not (token.isalpha() and len(token) < 2)
    ]


def stem(token: str) -> str:
    """Fold the common English inflections that matter for financial text."""

    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is", "as")):
        return token[:-1]
    if token.endswith("ing") and len(token) > 5 and not token.endswith(("eing", "oing", "ying")):
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("ly") and len(token) > 5:
        return token[:-2]
    if token.endswith("ity") and len(token) > 5:
        return token[:-3]
    return token


def _heading_lines(text: str) -> list[str]:
    headings = []
    for raw in text.splitlines()[:100]:
        line = raw.strip()
        if not line or len(line) > 90:
            continue
        words = line.split()
        if len(words) > 12:
            continue
        letters = [character for character in line if character.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(1 for character in letters if character.isupper()) / len(letters)
        if upper_ratio > 0.6 or line.istitle() or re.match(r"^(item|note|part)\s+[0-9ivx]+", line, flags=re.IGNORECASE):
            headings.append(line)
    return headings


_STATEMENT_TITLES: dict[str, tuple[str, ...]] = {
    "income": (
        "consolidated statement of income",
        "consolidated statements of income",
        "consolidated statement of operations",
        "consolidated statements of operations",
        "consolidated statement of earnings",
        "consolidated statements of earnings",
    ),
    "balance": (
        "consolidated balance sheet",
        "consolidated balance sheets",
        "consolidated statements of financial position",
    ),
    "cashflow": (
        "consolidated statement of cash flows",
        "consolidated statements of cash flows",
        "consolidated statement of cash flow",
    ),
}


def _statement_kinds(text: str) -> tuple[str, ...]:
    """Name the primary financial statements a page *is*.

    A page only counts when the statement title stands alone on one of its leading
    lines. Notes and index pages that mention a title inside a sentence or a table of
    contents entry are not the statement document.
    """

    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()][:12]
    return tuple(kind for kind, titles in _STATEMENT_TITLES.items() if any(line in titles for line in lines))


def _phrase_tokens(phrase: str) -> tuple[str, ...]:
    return tuple(stem(token) for token in _tokenize(phrase) if token not in _STEM_STOP)


def _concept_groups() -> dict[tuple[str, ...], tuple[tuple[tuple[str, ...], float], ...]]:
    groups: dict[tuple[str, ...], list[tuple[tuple[str, ...], float]]] = {}
    for group in _THESAURUS:
        members = [tokens for tokens in (_phrase_tokens(phrase) for phrase in group) if tokens]
        for member in members:
            group_targets = ((other, _EXPANSION_TERM_WEIGHT) for other in members if other != member)
            groups.setdefault(member, []).extend(group_targets)
    return {member: tuple(targets) for member, targets in groups.items()}


_CONCEPT_GROUPS = _concept_groups()


def _analyze_question(question: str) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]]:
    """Turn a natural-language question into weighted terms and concept phrases."""

    content = [token for token in _tokenize(question) if token not in _STEM_STOP]
    terms: dict[tuple[str, ...], float] = {}
    phrases: dict[tuple[str, ...], float] = {}
    for token in content:
        terms[(stem(token),)] = 1.0
    for size in (2, 3):
        for start in range(len(content) - size + 1):
            gram = tuple(stem(token) for token in content[start : start + size])
            terms[gram] = 1.0
            phrases[gram] = _PHRASE_WEIGHT
    for size in (1, 2, 3, 4):
        for start in range(len(content) - size + 1):
            gram = tuple(stem(token) for token in content[start : start + size])
            for target, weight in _CONCEPT_GROUPS.get(gram, ()):
                terms.setdefault(target, weight)
                if len(target) > 1:
                    phrases.setdefault(target, _EXPANSION_PHRASE_WEIGHT)
    return terms, phrases


def _statement_prior(question: str) -> dict[str, float]:
    lowered = _normalize_compounds(question).casefold()
    prior = {"income": 1.0, "balance": 1.0, "cashflow": 1.0}
    if re.search(r"\b(cash flows?|cashflow|capex|capital expenditures?|free cash flow)\b", lowered):
        prior["cashflow"] = 1.8
    if re.search(
        r"\b(balance sheet|liquidity|quick ratio|current ratio|working capital|current assets|current liabilities)\b",
        lowered,
    ):
        prior["balance"] = 1.8
    if re.search(
        r"\b(income statement|statement of operations|margin|gross profit|operating income|net income|revenues?|sales|"
        r"tax|ebitda|depreciation|amortization|profit)\b",
        lowered,
    ):
        prior["income"] = 1.45
    if re.search(r"capital[\s-]+intensive", lowered):
        return {"income": 1.5, "balance": 1.5, "cashflow": 1.5}
    return prior


def _statement_targets(question: str) -> set[str]:
    lowered = _normalize_compounds(question).casefold()
    if re.search(r"capital[\s-]+intensive", lowered):
        return {"income", "balance", "cashflow"}
    targets: set[str] = set()
    if re.search(
        r"\b(cash flows?|cashflow|capex|capital expenditures?|free cash flow|operating activities)\b", lowered
    ):
        targets.add("cashflow")
    if re.search(
        r"\b(balance sheet|liquidity|quick ratio|current ratio|working capital|current assets|current liabilities)\b",
        lowered,
    ):
        targets.add("balance")
    if re.search(
        r"\b(income statement|statement of operations|margin|gross profit|operating income|net income|revenues?|sales|"
        r"tax|ebitda|depreciation|amortization|profit)\b",
        lowered,
    ):
        targets.add("income")
    return targets


def _bm25(tf: int, length: int, average: float, k1: float = 1.35, b: float = 0.72) -> float:
    if not tf:
        return 0.0
    return tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * length / average))


def _contains_sequence(tokens: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    size = len(needle)
    first = needle[0]
    start = 0
    while True:
        try:
            index = tokens.index(first, start)
        except ValueError:
            return False
        if tokens[index : index + size] == needle:
            return True
        start = index + 1


def _rank_pages(
    pages: tuple[_IndexedPage, ...],
    analysis: tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], float]],
    question: str = "",
) -> list[_IndexedPage]:
    term_weights, phrase_weights = analysis
    total = len(pages)
    document_frequency: Counter[str] = Counter()
    for page in pages:
        document_frequency.update(page.token_set)
    average_length = sum(len(page.tokens) for page in pages) / total or 1.0

    def idf(token: str) -> float:
        df = document_frequency.get(token, 0)
        return math.log(1.0 + (total - df + 0.5) / (df + 0.5))

    singles: list[tuple[str, float, float]] = []
    for gram, weight in term_weights.items():
        if len(gram) != 1 or not document_frequency.get(gram[0]):
            continue
        singles.append((gram[0], weight, idf(gram[0])))
    multi: list[tuple[tuple[str, ...], float, float]] = []
    for gram, weight in phrase_weights.items():
        if len(gram) < 2 or not all(document_frequency.get(token) for token in gram):
            continue
        multi.append((gram, weight, sum(idf(token) for token in gram) / len(gram)))

    prior = _statement_prior(question)
    exhibit_start = _exhibit_index_start(pages)
    scored: list[tuple[float, int, _IndexedPage]] = []
    for page in pages:
        score = 0.0
        for token, weight, token_idf in singles:
            frequency = page.frequency.get(token, 0)
            if frequency:
                score += weight * token_idf * _bm25(frequency, len(page.tokens), average_length)
            heading = page.heading_frequency.get(token, 0)
            if heading:
                score += weight * token_idf * _HEADING_WEIGHT * _bm25(heading, page.heading_length, page.heading_length)
            title = page.title_frequency.get(token, 0)
            if title:
                score += weight * token_idf * _TITLE_WEIGHT * _bm25(title, page.title_length, page.title_length)
        for gram, weight, gram_idf in multi:
            if all(token in page.token_set for token in gram) and _contains_sequence(page.tokens, gram):
                score += weight * gram_idf
        if page.statement_kinds:
            score *= max(prior.get(kind, 1.0) for kind in page.statement_kinds)
        if page.head_text.casefold().startswith("exhibit"):
            score *= _EXHIBIT_PENALTY
        if page.page_number > exhibit_start.get(page.document_id, 0):
            score *= _EXHIBIT_PENALTY
        scored.append((score, -page.page_number, page))
    scored.sort(key=_score_order, reverse=True)

    targets = _statement_targets(question)
    top_score = scored[0][0] if scored else 0.0
    if targets and top_score > 0:
        # Primary statements are high-value evidence for metric questions. Inject only the
        # strongest matches so a multi-document collection cannot flood the result page.
        injectable = [
            (score, position, page)
            for score, position, page in scored
            if set(page.statement_kinds) & targets
        ]
        injected = {
            id(page) for _score, _position, page in sorted(injectable, key=_score_order, reverse=True)[:6]
        }
        if injected:
            boosted = [
                (
                    score + (_STATEMENT_INJECTION * top_score if id(page) in injected else 0.0),
                    position,
                    page,
                )
                for score, position, page in scored
            ]
            boosted.sort(key=_score_order, reverse=True)
            scored = boosted
    return [page for _score, _position, page in scored]


def _score_order(item: tuple[float, int, _IndexedPage]) -> tuple[float, int, str]:
    return item[0], item[1], item[2].document_id


def _exhibit_index_start(pages: tuple[_IndexedPage, ...]) -> dict[str, int]:
    """Find where each 10-K attachment section begins; exhibits are not evidence."""

    starts: dict[str, int] = {}
    for page in pages:
        lowered = page.text.casefold()
        if "exhibit index" in lowered or "exhibits filed" in lowered:
            starts[page.document_id] = max(starts.get(page.document_id, 0), page.page_number)
    return starts


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
