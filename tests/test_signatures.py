"""Tests for signature context extraction and packing."""

import ast
from dataclasses import replace

import pytest

from rac.context import build_context
from rac.types import Snippet

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")


def test_signatures_keep_interfaces_without_bodies(examples, tokenizer):
    example = replace(
        examples[0],
        import_statement="from pathlib import Path",
        context=[
            Snippet(
                "Reader",
                "reader.py",
                "class Reader:\n"
                "    @staticmethod\n"
                "    async def read(path: Path) -> str:\n"
                '        return "BODY_ONLY"\n'
                "\n"
                "def café(size: int = 2) -> int:\n"
                "    return size + 1\n",
            )
        ],
    )

    result = build_context(example, "signatures", 256, tokenizer)

    assert result.startswith("from pathlib import Path\n")
    assert "class Reader:\n    async def read(path: Path) -> str: ..." in result
    assert "def café(size: int = 2) -> int: ..." in result
    assert "BODY_ONLY" not in result
    assert "return size" not in result
    ast.parse(result)


def test_signatures_skip_oversized_stub_without_cutting_it(examples, tokenizer):
    example = replace(
        examples[0],
        import_statement="",
        context=[
            Snippet(
                "helpers",
                "helpers.py",
                "def large(a: int, b: int, c: int) -> int:\n"
                "    return a + b + c\n"
                "\n"
                "def small():\n"
                "    return 1\n",
            )
        ],
    )

    result = build_context(example, "signatures", 3, tokenizer)

    assert result == "def small(): ..."
    ast.parse(result)