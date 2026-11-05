"""Build compact Python interface stubs from repository candidates."""

from __future__ import annotations

from typing import Any

from rac.types import Example


def _attribute(node, source: bytes, indent: str) -> str | None:
    """Stub a class-level `name = value` or `name: type` statement without its value."""
    if node.type != "expression_statement" or not node.named_children:
        return None
    assignment = node.named_children[0]
    if assignment.type != "assignment":
        return None

    name = assignment.child_by_field_name("left")
    if name is None or name.type != "identifier":
        return None

    stub = indent + source[name.start_byte : name.end_byte].decode("utf-8")
    annotation = assignment.child_by_field_name("type")
    if annotation is not None:
        stub += ": " + source[annotation.start_byte : annotation.end_byte].decode("utf-8")
    if annotation is None or assignment.child_by_field_name("right") is not None:
        stub += " = ..."
    return stub


def _stub(node, source: bytes, indent: str = "") -> str | None:
    decorators = []
    if node.type == "decorated_definition":
        decorators = [child for child in node.children if child.type == "decorator"]
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
    lines = [source[decorator.start_byte : decorator.end_byte] for decorator in decorators]
    lines.append(source[node.start_byte : colon.end_byte])
    header = "\n".join(indent + line.decode("utf-8") for line in lines)
    if node.type == "function_definition":
        return f"{header} ..."

    members = []
    for child in body.named_children:
        member = _stub(child, source, indent + "    ")
        if member is None:
            member = _attribute(child, source, indent + "    ")
        if member is not None:
            members.append(member)

    if not members:
        return f"{header} ..."
    return f"{header}\n" + "\n".join(members)


def signature_map(example: Example, n: int, tokenizer: Any) -> str:
    """Return complete interface stubs within n model tokens.

    The example's import statements are left to build_prompt, which adds them after
    this block.
    """
    if n < 0:
        raise ValueError("Token budget must be non-negative.")
    if n == 0:
        return ""

    # Only the signatures method needs the optional parser dependencies.
    import tree_sitter_python
    from tree_sitter import Language, Parser

    parser = Parser(Language(tree_sitter_python.language()))
    units = []

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
        if len(tokenizer(combined, add_special_tokens=False)["input_ids"]) <= n:
            result = combined

    return result