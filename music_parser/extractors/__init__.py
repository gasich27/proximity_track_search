from music_parser.extractors.base_extractor import BaseExtractor
from music_parser.extractors.clap_extractor import CLAPExtractor
from music_parser.extractors.librosa_extractor import LibrosaExtractor
from music_parser.extractors.openl3_extractor import OpenL3Extractor
from music_parser.extractors.yamnet_extractor import YAMNetExtractor

__all__ = [
    "BaseExtractor",
    "CLAPExtractor",
    "LibrosaExtractor",
    "OpenL3Extractor",
    "YAMNetExtractor",
]


def build_extractors(names: list[str] | None = None) -> dict[str, BaseExtractor]:
    registry = {
        "librosa": LibrosaExtractor,
        "openl3": OpenL3Extractor,
        "yamnet": YAMNetExtractor,
        "clap": CLAPExtractor,
    }
    selected = names or list(registry.keys())
    return {name: registry[name]() for name in selected}
