"""Every docstring follows docs/agents/docstrings.md, so that marimo renders it.

marimo's documentation panel parses a docstring that has an Args, Returns, Raises or
Examples section with its own Google-style parser into Markdown tables, and renders any
other docstring as reStructuredText. These checks keep a docstring readable in both.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "src" / "bimpolars"

SECTION = re.compile(r"^( *)([A-Z][A-Za-z ]*):$")
KNOWN = {"Args", "Attributes", "Returns", "Raises", "Examples"}
LOOKALIKES = {"Arguments", "Parameters", "Return", "Yields", "Yield", "Example"}
ROLE = re.compile(r":[a-z]+(?::[a-z]+)*:`")
SINGLE_BACKTICK = re.compile(r"(?<!`)`(?!`)")
LITERAL = re.compile(r"``.+?``")
TYPED_ENTRY = re.compile(r"^\*{0,2}\w+ \([^)]+\): (?=\S)")
RETURN_TYPE = re.compile(r"^[\w.\[\], |]+: (?=\S)")
RAISES_ENTRY = re.compile(r"^[\w.]+: \S")


def _docstrings() -> Iterator[tuple[str, str, str]]:
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
                docstring = ast.get_docstring(node, clean=False)
                if docstring is not None:
                    name = getattr(node, "name", "module")
                    yield (
                        f"{path.relative_to(PACKAGE.parent)}:{node.body[0].lineno}",
                        name,
                        docstring,
                    )


def _problems(name: str, docstring: str) -> Iterator[str]:
    lines = docstring.splitlines()
    for line in lines:
        if ROLE.search(line):
            yield f"a Sphinx role, write ``name``: {line.strip()}"
        elif SINGLE_BACKTICK.search(line.replace("``", "")):
            yield f"single backticks, write ``code``: {line.strip()}"
        if any(line[m.end() : m.end() + 1].isalnum() for m in LITERAL.finditer(line)):
            yield f"a letter right after ``code`` breaks reStructuredText: {line.strip()}"

    section, indent, entries = "", 0, 0
    for line in lines:
        header = SECTION.match(line)
        if header and header.group(2) in KNOWN | LOOKALIKES:
            section, indent, entries = header.group(2), len(header.group(1)) + 4, 0
            if section in LOOKALIKES:
                yield f"marimo does not know the section {section}:"
            elif name == "__init__" and section != "Examples":
                yield f"{section}: in __init__, document the constructor in the class docstring"
            continue
        text = line.strip()
        if not section or not text:
            continue
        if len(line) - len(line.lstrip()) < indent:
            section = ""
            continue
        is_entry = len(line) - len(line.lstrip()) == indent
        description = text
        if section in {"Args", "Attributes"} and is_entry:
            if typed := TYPED_ENTRY.match(text):
                description = text[typed.end() :]
            else:
                yield f"{section} entry without `name (type): description`: {text}"
        if section == "Returns" and is_entry and entries == 0:
            if typed := RETURN_TYPE.match(text):
                description = text[typed.end() :]
            else:
                yield f"Returns without `Type: description` on its first line: {text}"
        if section in {"Args", "Attributes", "Returns"} and "|" in LITERAL.sub("", description):
            yield f"a | outside ``code`` splits marimo's table: {text}"
        if section == "Raises" and not (is_entry and RAISES_ENTRY.match(text)):
            yield f"a Raises entry over more than one line, marimo drops the rest: {text}"
        entries += is_entry


DOCSTRINGS = list(_docstrings())


@pytest.mark.parametrize(
    ("name", "docstring"),
    [(name, docstring) for _, name, docstring in DOCSTRINGS],
    ids=[f"{where} {name}" for where, name, _ in DOCSTRINGS],
)
def test_the_docstring_renders_in_marimo(name: str, docstring: str) -> None:
    problems = list(_problems(name, docstring))
    assert not problems, "\n".join(problems)
