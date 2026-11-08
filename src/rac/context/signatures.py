"""Build compact Python interface stubs from repository candidates."""

from __future__ import annotations

from typing import Any

from rac.context.tokens import count_tokens
from rac.types import Example


def _stub(node, source: bytes, indent: str = "") -> str | None:
    if node.type == "decorated_definition":
        node = node.child_by_field_name("definition")
        if node is None:
            return None

    if node.type not in {"function_definition", "class_definition"}:
        return None

    body = node.child_by_field_name("body")
    colon = next((child for child in node.children if child.type == ":"), None)
    if body is None or colon is None:
        return None

    # Tree-sitter offsets are byte offsets, including for Unicode identifiers.
    header = source[node.start_byte : colon.end_byte].decode("utf-8")
    if node.type == "function_definition":
        return f"{indent}{header} ..."

    members = []
    for child in body.named_children:
        member = _stub(child, source, indent + "    ")
        if member is not None:
            members.append(member)

    if not members:
        return f"{indent}{header} ..."
    return f"{indent}{header}\n" + "\n".join(members)


def signature_map(example: Example, n: int, tokenizer: Any) -> str:
    """Return imports and complete interface stubs within n model tokens."""
    if n < 0:
        raise ValueError("Token budget must be non-negative.")
    if n == 0:
        return ""

    # Only the signatures method needs the optional parser dependencies.
    import tree_sitter_python
    from tree_sitter import Language, Parser

    parser = Parser(Language(tree_sitter_python.language()))
    units = []

    imports = example.import_statement.encode("utf-8")
    for node in parser.parse(imports).root_node.named_children:
        if node.type in {"import_statement", "import_from_statement"} and not node.has_error:
            units.append(imports[node.start_byte : node.end_byte].decode("utf-8"))

    for candidate in example.context:
        source = candidate.snippet.encode("utf-8")
        for node in parser.parse(source).root_node.named_children:
            if not node.has_error:
                stub = _stub(node, source)
                if stub is not None:
                    units.append(stub)

    result = ""
    for unit in units:
        combined = f"{result}\n{unit}" if result else unit
        if count_tokens(combined, tokenizer) <= n:
            result = combined

    return result