"""Pinned Stanza and spaCy resource loading."""

import hashlib
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import spacy
import stanza
from pydantic import BaseModel, ConfigDict

from es_index_explorer.question_analysis.errors import MalformedInputError
from es_index_explorer.question_analysis.linguistics import (
    EntitySpan,
    ParsedSentence,
    ParsedWord,
)
from es_index_explorer.question_analysis.storage import fingerprint_file, sha256_file

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_RESOURCE_ROOT = PROJECT_ROOT / "artifacts" / "question_analysis" / "resources"
DEFAULT_STANZA_MODEL_DIR = DEFAULT_RESOURCE_ROOT / "stanza"
STANZA_MANIFEST_PATH = PACKAGE_ROOT / "resources" / "stanza-en-resource-manifest.json"
SPACY_MANIFEST_PATH = PACKAGE_ROOT / "resources" / "spacy-en-resource-manifest.json"


class ModelPin(BaseModel):
    """One selected Stanza package."""

    model_config = ConfigDict(frozen=True)

    package: str
    md5: str


class StanzaStaticManifest(BaseModel):
    """Static Stanza resource contract."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    language: str
    stanza_version: str
    processors: list[str]
    selected_models: dict[str, ModelPin]
    resources_url: str
    resources_sha256: str


class SpacyStaticManifest(BaseModel):
    """Static spaCy model contract."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    language: str
    model: str
    model_version: str
    pipeline_component: str
    spacy_compatibility: str
    wheel_url: str
    wheel_sha256: str


class _StanzaWord(Protocol):
    id: int
    text: str
    lemma: str | None
    upos: str | None
    xpos: str | None
    feats: str | None
    head: int
    deprel: str | None
    start_char: int | None
    end_char: int | None


class _StanzaSentence(Protocol):
    text: str
    words: list[_StanzaWord]


class _StanzaDocument(Protocol):
    sentences: list[_StanzaSentence]


class _StanzaPipeline(Protocol):
    def __call__(self, text: str) -> _StanzaDocument: ...


class _Entity(Protocol):
    text: str
    label_: str
    start_char: int
    end_char: int


class _SpacyDocument(Protocol):
    ents: tuple[_Entity, ...]


class _SpacyPipeline(Protocol):
    def __call__(self, text: str) -> _SpacyDocument: ...


@dataclass(frozen=True, slots=True)
class StanzaParser:
    """Verified Stanza English UD parser."""

    pipeline: _StanzaPipeline

    @classmethod
    def load(cls, model_dir: Path) -> "StanzaParser":
        manifest = load_stanza_manifest()
        verify_stanza_resources(model_dir, manifest)
        processors = {
            processor: manifest.selected_models[processor].package
            for processor in manifest.processors
        }
        pipeline = stanza.Pipeline(
            lang=manifest.language,
            dir=model_dir.as_posix(),
            processors=processors,
            package=None,
            download_method=None,
            use_gpu=False,
            verbose=False,
        )
        return cls(pipeline=cast(_StanzaPipeline, pipeline))

    def parse(self, text: str) -> tuple[ParsedSentence, ...]:
        document = self.pipeline(text)
        return tuple(
            ParsedSentence(
                sentence_index=sentence_index,
                text=sentence.text,
                words=tuple(
                    ParsedWord(
                        sentence_index=sentence_index,
                        word_id=int(word.id),
                        text=word.text,
                        lemma=word.lemma or "",
                        upos=word.upos or "",
                        xpos=word.xpos or "",
                        feats=_parse_feats(word.feats),
                        head=int(word.head),
                        deprel=word.deprel or "",
                        start_char=word.start_char,
                        end_char=word.end_char,
                    )
                    for word in sentence.words
                ),
            )
            for sentence_index, sentence in enumerate(document.sentences)
        )


@dataclass(frozen=True, slots=True)
class SpacyNer:
    """Verified spaCy NER-only pipeline."""

    pipeline: _SpacyPipeline

    @classmethod
    def load(cls) -> "SpacyNer":
        manifest = load_spacy_manifest()
        installed = version("en-core-web-sm")
        if installed != manifest.model_version:
            raise MalformedInputError(
                f"spaCy model version is {installed}; expected {manifest.model_version}"
            )
        pipeline = spacy.load(manifest.model, enable=[manifest.pipeline_component])
        return cls(pipeline=cast(_SpacyPipeline, pipeline))

    def parse(self, text: str) -> tuple[EntitySpan, ...]:
        document = self.pipeline(text)
        return tuple(
            EntitySpan(
                entity_index=index,
                text=entity.text,
                label=entity.label_,
                start_char=entity.start_char,
                end_char=entity.end_char,
            )
            for index, entity in enumerate(document.ents)
        )


def load_stanza_manifest() -> StanzaStaticManifest:
    return StanzaStaticManifest.model_validate_json(
        STANZA_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def load_spacy_manifest() -> SpacyStaticManifest:
    return SpacyStaticManifest.model_validate_json(
        SPACY_MANIFEST_PATH.read_text(encoding="utf-8")
    )


def download_stanza_resources(model_dir: Path) -> None:
    """Download the exact selected Stanza packages."""
    manifest = load_stanza_manifest()
    model_dir.mkdir(parents=True, exist_ok=True)
    stanza.download(
        lang=manifest.language,
        model_dir=model_dir.as_posix(),
        processors=None,
        package="default",
        resources_url=manifest.resources_url.rsplit("/", maxsplit=1)[0],
        resources_version=manifest.stanza_version,
        verbose=False,
    )
    verify_stanza_resources(model_dir, manifest)


def verify_stanza_resources(
    model_dir: Path, manifest: StanzaStaticManifest | None = None
) -> tuple[Path, ...]:
    """Verify all selected Stanza model files against the pinned catalogue."""
    manifest = manifest or load_stanza_manifest()
    resources_path = model_dir / "resources.json"
    if not resources_path.is_file():
        raise MalformedInputError(
            f"Missing Stanza resource catalogue {resources_path}; "
            "run features with --download-resources"
        )
    if sha256_file(resources_path) != manifest.resources_sha256:
        raise MalformedInputError("Stanza resource catalogue checksum mismatch")
    selected_paths: list[Path] = []
    for processor, pin in manifest.selected_models.items():
        path = model_dir / manifest.language / processor / f"{pin.package}.pt"
        if not path.is_file():
            raise MalformedInputError(
                f"Missing Stanza model {path}; run features with --download-resources"
            )
        with path.open("rb") as handle:
            actual_md5 = hashlib.file_digest(handle, "md5").hexdigest()
        if actual_md5 != pin.md5:
            raise MalformedInputError(
                f"Stanza model checksum mismatch for {path.name}"
            )
        selected_paths.append(path)
    return tuple(sorted(selected_paths))


def build_parser_resource_manifest(model_dir: Path) -> dict[str, object]:
    """Describe the exact parser files used by a feature run."""
    stanza_manifest = load_stanza_manifest()
    spacy_manifest = load_spacy_manifest()
    stanza_files = verify_stanza_resources(model_dir, stanza_manifest)
    spacy_package = import_module(spacy_manifest.model)
    package_file = Path(str(spacy_package.__file__)).resolve()
    package_root = package_file.parent
    spacy_files = tuple(
        path
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    return {
        "schema_version": 1,
        "stanza": {
            "distribution_version": version("stanza"),
            "language": stanza_manifest.language,
            "processors": stanza_manifest.processors,
            "resource_catalogue": fingerprint_file(
                model_dir / "resources.json", label="resources.json"
            ).model_dump(mode="json"),
            "static_manifest": fingerprint_file(
                STANZA_MANIFEST_PATH, label=STANZA_MANIFEST_PATH.name
            ).model_dump(mode="json"),
            "model_files": [
                _resource_file(path, model_dir) for path in stanza_files
            ],
        },
        "spacy": {
            "distribution_version": version("spacy"),
            "model": spacy_manifest.model,
            "model_version": version("en-core-web-sm"),
            "pipeline_component": spacy_manifest.pipeline_component,
            "static_manifest": fingerprint_file(
                SPACY_MANIFEST_PATH, label=SPACY_MANIFEST_PATH.name
            ).model_dump(mode="json"),
            "model_files": [
                _resource_file(path, package_root) for path in spacy_files
            ],
        },
    }


def _resource_file(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _parse_feats(value: str | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    pairs = []
    for item in value.split("|"):
        key, separator, feature_value = item.partition("=")
        if not separator:
            raise MalformedInputError(f"Malformed Stanza feature: {item}")
        pairs.append((key, feature_value))
    return tuple(sorted(pairs))
