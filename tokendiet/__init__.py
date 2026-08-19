"""Token-Diet — dynamic post-retrieval context compressor."""

import os as _os
import warnings as _warnings

# Set before huggingface_hub is imported anywhere downstream.
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:  # cosmetic only: SEC filings are XHTML, parsed with the HTML parser on purpose
    from bs4 import XMLParsedAsHTMLWarning

    _warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:  # noqa: BLE001
    pass

__version__ = "0.1.0"
