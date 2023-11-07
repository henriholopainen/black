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


def inject_line_range_placeholders_to_formatted_code(
    src_contents: str, dst_blocks: List[LinesBlock], line_range: Tuple[int, int]
) -> List[str]:
    dst_contents: List[str] = []

    start_ok, end_ok = False, False
    src_line_count = src_contents.count("\n")
    first_line_to_format = line_range[0]
    last_line_to_format = src_line_count - line_range[1]

    src_content_lines = [x.strip() for x in src_contents.split("\n")]
    standalone_comments = [
        x
        for x in src_content_lines
        if (
            x.startswith("#")
            or list(FMT_SKIP)[0][2:] in x
            or list(FMT_SKIP)[1][2:] in x
        )
    ]
    standalone_comment_line_numbers: List[int] = []

    standalone_comment_index = 0
    for src_content_line_index, src_content_line in enumerate(src_content_lines):
        if standalone_comment_index >= len(standalone_comments):
            break
        if src_content_line == standalone_comments[standalone_comment_index]:
            standalone_comment_line_numbers.append(src_content_line_index + 1)
            standalone_comment_index += 1

    dst_block_lines: List[List[str]] = []
    dst_block_last_line_numbers: List[int] = []

    for dst_block in dst_blocks:
        dst_block_line_numbers = [x.lineno for x in dst_block.original_line.leaves]
        last_line_number = max(dst_block_line_numbers)
        if dst_block.original_line.leaves[0].type == STANDALONE_COMMENT:
            # Standalone comment, possibly multiple lines
            last_line_number = (
                standalone_comment_line_numbers.pop(0)
                + dst_block.content_lines[0].count("\n")
                - 1
            )
        dst_block_last_line_numbers.append(last_line_number)
        dst_block_lines.append(dst_block.all_lines())
    dst_block_last_line_numbers.append(-1)
    dst_block_lines.append([])

    for index in range(len(dst_blocks)):
        block_lines = dst_block_lines[index]
        last_line_number_of_curr_block = dst_block_last_line_numbers[index]
        last_line_number_of_next_block = dst_block_last_line_numbers[index + 1]
        if (
            line_range
            and not start_ok
            and (
                first_line_to_format < last_line_number_of_curr_block
                or (
                    first_line_to_format == last_line_number_of_curr_block
                    and last_line_number_of_next_block != last_line_number_of_curr_block
                )
            )
        ):
            dst_contents.append(FORMAT_START)
            start_ok = True
        if (
            line_range
            and start_ok
            and not end_ok
            and (
                (last_line_to_format < last_line_number_of_curr_block)
                or (
                    last_line_to_format == last_line_number_of_curr_block
                    and last_line_number_of_next_block != last_line_number_of_curr_block
                )
            )
        ):
            if last_line_to_format == last_line_number_of_curr_block:
                dst_contents.extend(block_lines)
                block_lines.clear()
            else:
                consumed_empty_lines = 0
                is_standalone_comment_line = dst_block_lines[index][1].strip()[0] == "#"
                if is_standalone_comment_line:
                    dst_contents.append(block_lines[0])
                    comment_lines = block_lines[1].split("\n")
                    comment_lines_count = len(comment_lines) - 1
                    comment_lines_to_consume = last_line_to_format - (
                        last_line_number_of_curr_block - comment_lines_count
                    )
                    dst_contents.extend([
                        x + "\n" for x in comment_lines[:comment_lines_to_consume]
                    ])
                    block_lines[1] = "\n".join(comment_lines[comment_lines_to_consume:])
                    block_lines.pop(0)  # pop the block_lines[0] append
                else:
                    for block_line in block_lines:
                        if not block_line.strip():
                            dst_contents.append(block_line)
                            consumed_empty_lines += 1
                        else:
                            break
                    [block_lines.pop(0) for _ in range(consumed_empty_lines)]
            dst_contents.append(FORMAT_END)
            end_ok = True
        dst_contents.extend(block_lines)

    return dst_contents


def combine_format_changes_to_source(
    src_contents: str, dst_contents_list: List[str], line_range: Tuple[int, int]
) -> str:
    dst_contents = "".join(dst_contents_list)
    dst_formatted_start_index = dst_contents.index(FORMAT_START)
    dst_formatted_end_index = dst_contents.index(FORMAT_END)
    dst_contents_substr = dst_contents[
        dst_formatted_start_index + len(FORMAT_START) : dst_formatted_end_index
    ]

    src_line_breaks = [0] + [m.start() + 1 for m in re.finditer(r"\n", src_contents)]
    format_start_line = line_range[0] - 1
    format_end_line = line_range[1] + 1
    index_of_line_before_format_start = src_line_breaks[format_start_line]
    index_of_line_after_format_end = src_line_breaks[-format_end_line]
    src_before_lines = src_contents[:index_of_line_before_format_start]
    src_after_lines = src_contents[index_of_line_after_format_end:]

    return src_before_lines + dst_contents_substr + src_after_lines
