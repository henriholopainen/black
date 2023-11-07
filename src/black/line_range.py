from typing import Optional, Tuple

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

    start_line, end_line = lines

    # Find the first line that contains start_line after a line break
    for node in src_node.post_order():
        if not node or node.type in [token.INDENT, token.DEDENT]:
            continue
        node_start, node_end = line_span(node)
        if node_start <= start_line and node_end >= start_line:
            if node.type == token.NEWLINE:
                start_line = node_start + 1
                break
            curr = node
            prev = preceding_leaf(curr)
            while prev and prev.type != token.NEWLINE:
                curr = prev
                prev = preceding_leaf(curr)
            if curr:
                start_line = curr.get_lineno() or start_line
            break

    # Find the last line that contains end_line before a line break
    for node in src_node.post_order():
        if not node or node.type in [token.INDENT, token.DEDENT]:
            continue
        node_start, node_end = line_span(node)
        if node_start <= end_line and node_end >= end_line:
            if node.type == token.NEWLINE:
                end_line = node_end
                break
            curr = node
            next = following_leaf(curr)
            while next and next.type != token.NEWLINE:
                curr = next
                next = following_leaf(curr)
            if next:
                end_line = next.get_lineno() or end_line
            break

    # Extend to cover neighbouring empty lines and FMT_OFF/FMT_ON comments
    while (
        start_line > 1
        and src_line_empty[start_line - 2]
        or src_lines[start_line - 2].strip() in FMT_OFF
    ):
        start_line -= 1
    while (
        end_line < lines_in_file
        and src_line_empty[end_line]
        or src_lines[end_line].strip() in FMT_ON
    ):
        end_line += 1

    if mode.preview:
        # Preview mode currently moves ellipsis to the same line as block open
        # if there's nothing between the comma and ellipsis
        if (
            end_line > 1
            and end_line < lines_in_file - 1
            and "..." == src_lines[end_line].strip()
            and ":" == src_lines[end_line - 1][-1].strip()
        ):
            end_line += 1
        if (
            start_line > 1
            and start_line < lines_in_file - 1
            and "..." == src_lines[start_line].strip()
            and ":" == src_lines[start_line - 1].strip()
        ):
            start_line -= 1
        pass

    # To keep --lines stable, we count the end index as lines from EOF
    return (start_line, lines_in_file - end_line)
