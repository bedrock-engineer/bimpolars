# Docstrings

The API is read in marimo's documentation panel, so a docstring is written for that renderer.
marimo parses a docstring with an `Args:`, `Returns:`, `Raises:` or `Examples:` section with
its own small Google-style parser into Markdown tables, and renders any other docstring as
reStructuredText. `tests/test_docstrings.py` checks the shape below on every docstring.

- Inline code goes in double backticks, ``` ``name`` ```, which is code in Markdown and in
  reStructuredText alike. Another function or constant is named the same way:
  ``` ``bimpolars.set_crs`` ```, ``` ``CRS_COLUMN`` ```. A space or punctuation follows the
  closing backticks (``` ``mesh`` columns ```, not a plural glued on). A link is the bare URL
  in brackets.
- `Args:` and `Attributes:` entries carry the type: `name (type): description`. marimo fills
  an empty type cell from the signature for plain functions only, and an empty cell next to a
  description with code breaks its table.
- `Returns:` starts with the type: `BimDataFrame: The model as a BimDataFrame.` A long
  annotation shortens to its outer type (`tuple`, `np.ndarray`). A generator documents
  `Returns:` too (`Iterator[tuple]: ...`), as marimo has no `Yields:`.
- Each `Raises:` entry is one line, as marimo keeps only the first line of an entry: an
  exception is listed once per condition, and the reasoning goes into the description.
- A class documents its constructor in the class docstring, `Args:` and `Raises:` beside
  `Attributes:`, and `__init__` has no docstring.
- A `|` in an `Args:`, `Attributes:` or `Returns:` description stays inside ``` ``code`` ```,
  as a bare one splits marimo's table.

A class docstring with only `Attributes:` (the NamedTuples) still renders as
reStructuredText, its attributes run together: marimo's parser does not look for that
section on its own. That is marimo's limit, and the docstring stays as above.
