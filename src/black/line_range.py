import re
from typing import List, Optional, Tuple

from black.comments import FMT_OFF, FMT_ON, FMT_SKIP
from black.lines import LinesBlock
from black.mode import Mode
from black.nodes import STANDALONE_COMMENT, following_leaf, line_span, preceding_leaf
from black.parsing import lib2to3_parse
from blib2to3.pgen2 import token
from blib2to3.pytree import Node

FORMAT_START = "# _______BLACK______FORMAT_______START_______"
FORMAT_END = "# _______BLACK______FORMAT_______END_______"


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


def calculate_line_range(
    lines: Tuple[int, int],
    src_contents: str,
    mode: Mode,
    src_node_input: Optional[Node] = None,  # Optimisation for tests
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


def inject_line_range_placeholders_to_formatted_code(
    src_contents: str, dst_blocks: List[LinesBlock], line_range: Tuple[int, int]
) -> List[str]:
    """Inject FORMAT_START and FORMAT_END to formatted code so that the markers can be
    used to extract the formatted lines corresponding to line_range in source"""
    dst_contents: List[str] = []

    # Because we don't track standalone comments' line numbers in nodes, let's find
    # them from the source
    standalone_comment_line_numbers = [
        index + 1  # Adjust to line_span range [1, line_count]
        for index, line in enumerate([x.strip() for x in src_contents.split("\n")])
        if (
            line.startswith("#")
            # TODO: this should be revisited if PR #3978 gets accepted
            or any(line.strip().endswith(skip) for skip in FMT_SKIP)
        )
    ]

    dst_block_lines: List[List[str]] = []
    dst_block_last_line_numbers: List[int] = []

    for block in dst_blocks:
        dst_block_line_numbers = [x.lineno for x in block.original_line.leaves]
        for leaf in block.original_line.leaves:
            if leaf.type == STANDALONE_COMMENT:
                # Standalone comments possibly span multiple lines
                # If we have multiple identical comments, pick the later
                content_line_index = [
                    line.strip() for line in reversed(block.content_lines)
                ].index(leaf.value) + 1
                dst_block_line_numbers.append(
                    standalone_comment_line_numbers.pop(0)
                    + block.content_lines[-content_line_index].count("\n")
                    - 1
                )
        dst_block_last_line_numbers.append(max(dst_block_line_numbers))
        dst_block_lines.append(block.all_lines())

    # Avoid checking for bounds with faux elements
    dst_block_last_line_numbers.append(-1)
    dst_block_lines.append([])

    start_marked, end_marked = False, False
    first_line_to_format = line_range[0]
    last_line_to_format = src_contents.count("\n") - line_range[1]

    for index in range(len(dst_blocks)):
        block_lines = dst_block_lines[index]
        last_line_number_of_curr_block = dst_block_last_line_numbers[index]
        last_line_number_of_next_block = dst_block_last_line_numbers[index + 1]

        first_line_passed = first_line_to_format < last_line_number_of_curr_block
        on_the_first_line_but_next_is_past = (
            first_line_to_format == last_line_number_of_curr_block
            and last_line_number_of_next_block != last_line_number_of_curr_block
        )

        last_line_passed = last_line_to_format < last_line_number_of_curr_block
        on_the_last_line_but_next_is_past = (
            last_line_to_format == last_line_number_of_curr_block
            and last_line_number_of_next_block != last_line_number_of_curr_block
        )

        if not start_marked and (
            first_line_passed or on_the_first_line_but_next_is_past
        ):
            dst_contents.append(FORMAT_START)
            start_marked = True
        if (
            start_marked
            and not end_marked
            and (last_line_passed or on_the_last_line_but_next_is_past)
        ):
            if last_line_to_format == last_line_number_of_curr_block:
                # We are at the end of our last line, so everything goes before marker
                dst_contents.extend(block_lines)
                dst_contents.append(FORMAT_END)
                end_marked = True
                continue

            empty_lines_appended = False
            if not block_lines[0].strip():
                # Add empty lines from the formatting before marker
                dst_contents.append(block_lines[0])
                empty_lines_appended = True

            is_fmt_off_line = block_lines[1].strip()[0] == "#"

            if is_fmt_off_line:
                # We know that this line is a fmt:off line and it will span multiple
                # lines in the source because true standalone comments are only one
                # line, and that is handled above. Add necessary amount of lines
                # before marker
                comment_lines = block_lines[1].split("\n")
                comment_lines_count = len(comment_lines) - 1
                comment_lines_to_consume = last_line_to_format - (
                    last_line_number_of_curr_block - comment_lines_count
                )
                dst_contents.extend([
                    x + "\n" for x in comment_lines[:comment_lines_to_consume]
                ])
                block_lines[1] = "\n".join(comment_lines[comment_lines_to_consume:])

            dst_contents.append(FORMAT_END)
            end_marked = True
            if empty_lines_appended:
                # Pop so we won't append the empty lines twice
                block_lines.pop(0)

        dst_contents.extend(block_lines)

    return dst_contents


def combine_format_changes_to_source(
    src_contents: str, dst_contents: str, line_range: Tuple[int, int]
) -> str:
    """Combine start and end from src_contents with formatted lines marked by
    FORMAT_START and FORMAT_END in dst_contents"""
    formatted_changes = dst_contents[
        dst_contents.index(FORMAT_START)
        + len(FORMAT_START) : dst_contents.index(FORMAT_END)
    ]

    src_line_break_locations = [0] + [
        m.start() + 1 for m in re.finditer(r"\n", src_contents)
    ]
    unformatted_before = src_contents[: src_line_break_locations[line_range[0] - 1]]
    unformatted_after = src_contents[src_line_break_locations[-(line_range[1] + 1)] :]

    return unformatted_before + formatted_changes + unformatted_after
