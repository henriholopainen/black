from typing import List, Optional, Tuple

from black.comments import FMT_OFF, FMT_ON
from black.mode import Mode
from black.nodes import following_leaf, line_span, preceding_leaf
from black.parsing import lib2to3_parse
from blib2to3.pgen2 import token
from blib2to3.pytree import Node


def _validate_line_range_input(lines: Tuple[int, int], lines_in_file: int) -> None:
    if lines[1] > lines_in_file:
        raise ValueError(f"Invalid --lines (source is {lines_in_file} lines).")
    if lines[0] > lines[1]:
        raise ValueError("Invalid --lines (start must be smaller than end).")
    if lines[0] < 1 or lines[1] < 1:
        raise ValueError("Invalid --lines (start and end must be >0).")


def _find_first_spanning_line(src_node: Node, line_number: int) -> int:
    """Return the line number that starts the expression containing line_number"""
    for node in src_node.post_order():
        if not node or node.type in [token.INDENT, token.DEDENT]:
            continue
        node_start, node_end = line_span(node)
        if node_start <= line_number and node_end >= line_number:
            if node.type == token.NEWLINE:
                line_number = node_start + 1
                break
            curr = node
            prev = preceding_leaf(curr)
            while prev and prev.type != token.NEWLINE:
                curr = prev
                prev = preceding_leaf(curr)
            if curr:
                line_number = curr.get_lineno() or line_number
            break
    return line_number


def _find_last_spanning_line(src_node: Node, line_number: int) -> int:
    """Return the line number that ends the expression containing line_number"""
    for node in src_node.post_order():
        if not node or node.type in [token.INDENT, token.DEDENT]:
            continue
        node_start, node_end = line_span(node)
        if node_start <= line_number and node_end >= line_number:
            if node.type == token.NEWLINE:
                line_number = node_end
                break
            curr = node
            next = following_leaf(curr)
            while next and next.type != token.NEWLINE:
                curr = next
                next = following_leaf(curr)
            if next:
                line_number = next.get_lineno() or line_number
            break
    return line_number


def _adjust_start(
    line: int,
    src_lines: List[str],
    src_line_empty: List[bool],
    lines_in_file: int,
    preview: bool,
) -> int:
    """Adjust line number so that empty lines and fmt:off are made part of our span"""
    while (
        line > 1 and src_line_empty[line - 2] or src_lines[line - 2].strip() in FMT_OFF
    ):
        line -= 1

    # Preview mode currently moves ellipsis to the same line as block open
    # if there's nothing between the comma and ellipsis, so adjust for that
    if (
        preview
        and line > 1
        and line < lines_in_file - 1
        and "..." == src_lines[line].strip()
        and ":" == src_lines[line - 1].strip()
    ):
        line -= 1

    return line


def _adjust_end(
    line: int,
    src_lines: List[str],
    src_line_empty: List[bool],
    lines_in_file: int,
    preview: bool,
) -> int:
    """Adjust line number so that empty lines and fmt:on are made part of our span"""
    while (
        line < lines_in_file
        and src_line_empty[line]
        or src_lines[line].strip() in FMT_ON
    ):
        line += 1

    # Preview mode currently moves ellipsis to the same line as block open
    # if there's nothing between the comma and ellipsis, so adjust for that
    if (
        preview
        and line > 1
        and line < lines_in_file - 1
        and "..." == src_lines[line].strip()
        and ":" == src_lines[line - 1][-1].strip()
    ):
        line += 1

    return line


def calculate_line_range(  # noqa: C901
    lines: Tuple[int, int],
    src_contents: str,
    mode: Mode,
    src_node_input: Optional[Node] = None,
) -> Tuple[int, int]:
    lines_in_file = src_contents.count("\n")
    src_lines = src_contents.split("\n")
    src_line_empty = [not x.strip() for x in src_lines]

    _validate_line_range_input(lines, lines_in_file)

    src_node = (
        lib2to3_parse(src_contents.lstrip(), mode.target_versions)
        if not src_node_input
        else src_node_input
    )

    start_line = _adjust_start(
        _find_first_spanning_line(src_node, lines[0]),
        src_lines,
        src_line_empty,
        lines_in_file,
        mode.preview,
    )
    end_line = _adjust_end(
        _find_last_spanning_line(src_node, lines[1]),
        src_lines,
        src_line_empty,
        lines_in_file,
        mode.preview,
    )

    # To keep --lines stable over iterations, we count the end index as lines from EOF
    return (start_line, lines_in_file - end_line)
