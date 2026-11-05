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

    assert "from pathlib import Path" not in result
    assert "class Reader:\n    @staticmethod\n    async def read(path: Path) -> str: ..." in result
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


def test_signatures_spend_no_budget_on_imports(examples, tokenizer):
    # build_prompt adds the import statements after this block, so repeating them here
    # would charge them to the budget twice.
    example = replace(
        examples[0],
        import_statement="from shop.models import Item",
        context=[Snippet("small", "helpers.py", "def small():\n    return 1\n")],
    )

    result = build_context(example, "signatures", 4, tokenizer)

    assert result == "def small(): ..."


def test_signatures_keep_decorators(examples, tokenizer):
    # A stub without its decorator misstates the interface: a property reads as a method.
    example = replace(
        examples[0],
        import_statement="",
        context=[
            Snippet(
                "Box",
                "box.py",
                "@dataclass(frozen=True)\n"
                "class Box:\n"
                "    @property\n"
                "    def area(self) -> int:\n"
                "        return self.w * self.h\n"
                "\n"
                "@lru_cache\n"
                "@validate\n"
                "def load(path: str) -> Box:\n"
                "    return Box()\n",
            )
        ],
    )

    result = build_context(example, "signatures", 256, tokenizer)

    assert result == (
        "@dataclass(frozen=True)\n"
        "class Box:\n"
        "    @property\n"
        "    def area(self) -> int: ...\n"
        "@lru_cache\n"
        "@validate\n"
        "def load(path: str) -> Box: ..."
    )
    ast.parse(result)
