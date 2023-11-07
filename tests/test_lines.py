from dataclasses import replace

from black import Mode, calculate_line_range, format_str
from black.parsing import lib2to3_parse
from tests.util import read_data_with_mode

# fmt:off
expected_line_ranges = {
    (1, 1): (1, 39), (1, 2): (1, 35), (1, 3): (1, 35), (1, 4): (1, 35), (1, 5): (1, 35), (1, 6): (1, 33), (1, 7): (1, 33), (1, 8): (1, 30), (1, 9): (1, 30), (1, 10): (1, 30), (1, 11): (1, 29), (1, 12): (1, 28), (1, 13): (1, 27), (1, 14): (1, 26), (1, 15): (1, 25), (1, 16): (1, 23), (1, 17): (1, 23), (1, 18): (1, 22), (1, 19): (1, 21), (1, 20): (1, 20), (1, 21): (1, 19), (1, 22): (1, 18), (1, 23): (1, 17), (1, 24): (1, 16), (1, 25): (1, 15), (1, 26): (1, 14), (1, 27): (1, 13), (1, 28): (1, 12), (1, 29): (1, 11), (1, 30): (1, 5), (1, 31): (1, 5), (1, 32): (1, 5), (1, 33): (1, 5), (1, 34): (1, 5), (1, 35): (1, 5), (1, 36): (1, 0), (1, 37): (1, 0), (1, 38): (1, 0), (1, 39): (1, 0), (1, 40): (1, 0),  # noqa: B950
    (2, 2): (2, 35), (2, 3): (2, 35), (2, 4): (2, 35), (2, 5): (2, 35), (2, 6): (2, 33), (2, 7): (2, 33), (2, 8): (2, 30), (2, 9): (2, 30), (2, 10): (2, 30), (2, 11): (2, 29), (2, 12): (2, 28), (2, 13): (2, 27), (2, 14): (2, 26), (2, 15): (2, 25), (2, 16): (2, 23), (2, 17): (2, 23), (2, 18): (2, 22), (2, 19): (2, 21), (2, 20): (2, 20), (2, 21): (2, 19), (2, 22): (2, 18), (2, 23): (2, 17), (2, 24): (2, 16), (2, 25): (2, 15), (2, 26): (2, 14), (2, 27): (2, 13), (2, 28): (2, 12), (2, 29): (2, 11), (2, 30): (2, 5), (2, 31): (2, 5), (2, 32): (2, 5), (2, 33): (2, 5), (2, 34): (2, 5), (2, 35): (2, 5), (2, 36): (2, 0), (2, 37): (2, 0), (2, 38): (2, 0), (2, 39): (2, 0), (2, 40): (2, 0),  # noqa: B950
    (3, 3): (2, 35), (3, 4): (2, 35), (3, 5): (2, 35), (3, 6): (2, 33), (3, 7): (2, 33), (3, 8): (2, 30), (3, 9): (2, 30), (3, 10): (2, 30), (3, 11): (2, 29), (3, 12): (2, 28), (3, 13): (2, 27), (3, 14): (2, 26), (3, 15): (2, 25), (3, 16): (2, 23), (3, 17): (2, 23), (3, 18): (2, 22), (3, 19): (2, 21), (3, 20): (2, 20), (3, 21): (2, 19), (3, 22): (2, 18), (3, 23): (2, 17), (3, 24): (2, 16), (3, 25): (2, 15), (3, 26): (2, 14), (3, 27): (2, 13), (3, 28): (2, 12), (3, 29): (2, 11), (3, 30): (2, 5), (3, 31): (2, 5), (3, 32): (2, 5), (3, 33): (2, 5), (3, 34): (2, 5), (3, 35): (2, 5), (3, 36): (2, 0), (3, 37): (2, 0), (3, 38): (2, 0), (3, 39): (2, 0), (3, 40): (2, 0),  # noqa: B950
    (4, 4): (2, 35), (4, 5): (2, 35), (4, 6): (2, 33), (4, 7): (2, 33), (4, 8): (2, 30), (4, 9): (2, 30), (4, 10): (2, 30), (4, 11): (2, 29), (4, 12): (2, 28), (4, 13): (2, 27), (4, 14): (2, 26), (4, 15): (2, 25), (4, 16): (2, 23), (4, 17): (2, 23), (4, 18): (2, 22), (4, 19): (2, 21), (4, 20): (2, 20), (4, 21): (2, 19), (4, 22): (2, 18), (4, 23): (2, 17), (4, 24): (2, 16), (4, 25): (2, 15), (4, 26): (2, 14), (4, 27): (2, 13), (4, 28): (2, 12), (4, 29): (2, 11), (4, 30): (2, 5), (4, 31): (2, 5), (4, 32): (2, 5), (4, 33): (2, 5), (4, 34): (2, 5), (4, 35): (2, 5), (4, 36): (2, 0), (4, 37): (2, 0), (4, 38): (2, 0), (4, 39): (2, 0), (4, 40): (2, 0),  # noqa: B950
    (5, 5): (5, 35), (5, 6): (5, 33), (5, 7): (5, 33), (5, 8): (5, 30), (5, 9): (5, 30), (5, 10): (5, 30), (5, 11): (5, 29), (5, 12): (5, 28), (5, 13): (5, 27), (5, 14): (5, 26), (5, 15): (5, 25), (5, 16): (5, 23), (5, 17): (5, 23), (5, 18): (5, 22), (5, 19): (5, 21), (5, 20): (5, 20), (5, 21): (5, 19), (5, 22): (5, 18), (5, 23): (5, 17), (5, 24): (5, 16), (5, 25): (5, 15), (5, 26): (5, 14), (5, 27): (5, 13), (5, 28): (5, 12), (5, 29): (5, 11), (5, 30): (5, 5), (5, 31): (5, 5), (5, 32): (5, 5), (5, 33): (5, 5), (5, 34): (5, 5), (5, 35): (5, 5), (5, 36): (5, 0), (5, 37): (5, 0), (5, 38): (5, 0), (5, 39): (5, 0), (5, 40): (5, 0),  # noqa: B950
    (6, 6): (5, 33), (6, 7): (5, 33), (6, 8): (5, 30), (6, 9): (5, 30), (6, 10): (5, 30), (6, 11): (5, 29), (6, 12): (5, 28), (6, 13): (5, 27), (6, 14): (5, 26), (6, 15): (5, 25), (6, 16): (5, 23), (6, 17): (5, 23), (6, 18): (5, 22), (6, 19): (5, 21), (6, 20): (5, 20), (6, 21): (5, 19), (6, 22): (5, 18), (6, 23): (5, 17), (6, 24): (5, 16), (6, 25): (5, 15), (6, 26): (5, 14), (6, 27): (5, 13), (6, 28): (5, 12), (6, 29): (5, 11), (6, 30): (5, 5), (6, 31): (5, 5), (6, 32): (5, 5), (6, 33): (5, 5), (6, 34): (5, 5), (6, 35): (5, 5), (6, 36): (5, 0), (6, 37): (5, 0), (6, 38): (5, 0), (6, 39): (5, 0), (6, 40): (5, 0),  # noqa: B950
    (7, 7): (7, 33), (7, 8): (7, 30), (7, 9): (7, 30), (7, 10): (7, 30), (7, 11): (7, 29), (7, 12): (7, 28), (7, 13): (7, 27), (7, 14): (7, 26), (7, 15): (7, 25), (7, 16): (7, 23), (7, 17): (7, 23), (7, 18): (7, 22), (7, 19): (7, 21), (7, 20): (7, 20), (7, 21): (7, 19), (7, 22): (7, 18), (7, 23): (7, 17), (7, 24): (7, 16), (7, 25): (7, 15), (7, 26): (7, 14), (7, 27): (7, 13), (7, 28): (7, 12), (7, 29): (7, 11), (7, 30): (7, 5), (7, 31): (7, 5), (7, 32): (7, 5), (7, 33): (7, 5), (7, 34): (7, 5), (7, 35): (7, 5), (7, 36): (7, 0), (7, 37): (7, 0), (7, 38): (7, 0), (7, 39): (7, 0), (7, 40): (7, 0),  # noqa: B950
    (8, 8): (7, 30), (8, 9): (7, 30), (8, 10): (7, 30), (8, 11): (7, 29), (8, 12): (7, 28), (8, 13): (7, 27), (8, 14): (7, 26), (8, 15): (7, 25), (8, 16): (7, 23), (8, 17): (7, 23), (8, 18): (7, 22), (8, 19): (7, 21), (8, 20): (7, 20), (8, 21): (7, 19), (8, 22): (7, 18), (8, 23): (7, 17), (8, 24): (7, 16), (8, 25): (7, 15), (8, 26): (7, 14), (8, 27): (7, 13), (8, 28): (7, 12), (8, 29): (7, 11), (8, 30): (7, 5), (8, 31): (7, 5), (8, 32): (7, 5), (8, 33): (7, 5), (8, 34): (7, 5), (8, 35): (7, 5), (8, 36): (7, 0), (8, 37): (7, 0), (8, 38): (7, 0), (8, 39): (7, 0), (8, 40): (7, 0),  # noqa: B950
    (9, 9): (7, 30), (9, 10): (7, 30), (9, 11): (7, 29), (9, 12): (7, 28), (9, 13): (7, 27), (9, 14): (7, 26), (9, 15): (7, 25), (9, 16): (7, 23), (9, 17): (7, 23), (9, 18): (7, 22), (9, 19): (7, 21), (9, 20): (7, 20), (9, 21): (7, 19), (9, 22): (7, 18), (9, 23): (7, 17), (9, 24): (7, 16), (9, 25): (7, 15), (9, 26): (7, 14), (9, 27): (7, 13), (9, 28): (7, 12), (9, 29): (7, 11), (9, 30): (7, 5), (9, 31): (7, 5), (9, 32): (7, 5), (9, 33): (7, 5), (9, 34): (7, 5), (9, 35): (7, 5), (9, 36): (7, 0), (9, 37): (7, 0), (9, 38): (7, 0), (9, 39): (7, 0), (9, 40): (7, 0),  # noqa: B950
    (10, 10): (7, 30), (10, 11): (7, 29), (10, 12): (7, 28), (10, 13): (7, 27), (10, 14): (7, 26), (10, 15): (7, 25), (10, 16): (7, 23), (10, 17): (7, 23), (10, 18): (7, 22), (10, 19): (7, 21), (10, 20): (7, 20), (10, 21): (7, 19), (10, 22): (7, 18), (10, 23): (7, 17), (10, 24): (7, 16), (10, 25): (7, 15), (10, 26): (7, 14), (10, 27): (7, 13), (10, 28): (7, 12), (10, 29): (7, 11), (10, 30): (7, 5), (10, 31): (7, 5), (10, 32): (7, 5), (10, 33): (7, 5), (10, 34): (7, 5), (10, 35): (7, 5), (10, 36): (7, 0), (10, 37): (7, 0), (10, 38): (7, 0), (10, 39): (7, 0), (10, 40): (7, 0),  # noqa: B950
    (11, 11): (11, 29), (11, 12): (11, 28), (11, 13): (11, 27), (11, 14): (11, 26), (11, 15): (11, 25), (11, 16): (11, 23), (11, 17): (11, 23), (11, 18): (11, 22), (11, 19): (11, 21), (11, 20): (11, 20), (11, 21): (11, 19), (11, 22): (11, 18), (11, 23): (11, 17), (11, 24): (11, 16), (11, 25): (11, 15), (11, 26): (11, 14), (11, 27): (11, 13), (11, 28): (11, 12), (11, 29): (11, 11), (11, 30): (11, 5), (11, 31): (11, 5), (11, 32): (11, 5), (11, 33): (11, 5), (11, 34): (11, 5), (11, 35): (11, 5), (11, 36): (11, 0), (11, 37): (11, 0), (11, 38): (11, 0), (11, 39): (11, 0), (11, 40): (11, 0),  # noqa: B950
    (12, 12): (12, 28), (12, 13): (12, 27), (12, 14): (12, 26), (12, 15): (12, 25), (12, 16): (12, 23), (12, 17): (12, 23), (12, 18): (12, 22), (12, 19): (12, 21), (12, 20): (12, 20), (12, 21): (12, 19), (12, 22): (12, 18), (12, 23): (12, 17), (12, 24): (12, 16), (12, 25): (12, 15), (12, 26): (12, 14), (12, 27): (12, 13), (12, 28): (12, 12), (12, 29): (12, 11), (12, 30): (12, 5), (12, 31): (12, 5), (12, 32): (12, 5), (12, 33): (12, 5), (12, 34): (12, 5), (12, 35): (12, 5), (12, 36): (12, 0), (12, 37): (12, 0), (12, 38): (12, 0), (12, 39): (12, 0), (12, 40): (12, 0),  # noqa: B950
    (13, 13): (13, 27), (13, 14): (13, 26), (13, 15): (13, 25), (13, 16): (13, 23), (13, 17): (13, 23), (13, 18): (13, 22), (13, 19): (13, 21), (13, 20): (13, 20), (13, 21): (13, 19), (13, 22): (13, 18), (13, 23): (13, 17), (13, 24): (13, 16), (13, 25): (13, 15), (13, 26): (13, 14), (13, 27): (13, 13), (13, 28): (13, 12), (13, 29): (13, 11), (13, 30): (13, 5), (13, 31): (13, 5), (13, 32): (13, 5), (13, 33): (13, 5), (13, 34): (13, 5), (13, 35): (13, 5), (13, 36): (13, 0), (13, 37): (13, 0), (13, 38): (13, 0), (13, 39): (13, 0), (13, 40): (13, 0),  # noqa: B950
    (14, 14): (14, 26), (14, 15): (14, 25), (14, 16): (14, 23), (14, 17): (14, 23), (14, 18): (14, 22), (14, 19): (14, 21), (14, 20): (14, 20), (14, 21): (14, 19), (14, 22): (14, 18), (14, 23): (14, 17), (14, 24): (14, 16), (14, 25): (14, 15), (14, 26): (14, 14), (14, 27): (14, 13), (14, 28): (14, 12), (14, 29): (14, 11), (14, 30): (14, 5), (14, 31): (14, 5), (14, 32): (14, 5), (14, 33): (14, 5), (14, 34): (14, 5), (14, 35): (14, 5), (14, 36): (14, 0), (14, 37): (14, 0), (14, 38): (14, 0), (14, 39): (14, 0), (14, 40): (14, 0),  # noqa: B950
    (15, 15): (15, 25), (15, 16): (15, 23), (15, 17): (15, 23), (15, 18): (15, 22), (15, 19): (15, 21), (15, 20): (15, 20), (15, 21): (15, 19), (15, 22): (15, 18), (15, 23): (15, 17), (15, 24): (15, 16), (15, 25): (15, 15), (15, 26): (15, 14), (15, 27): (15, 13), (15, 28): (15, 12), (15, 29): (15, 11), (15, 30): (15, 5), (15, 31): (15, 5), (15, 32): (15, 5), (15, 33): (15, 5), (15, 34): (15, 5), (15, 35): (15, 5), (15, 36): (15, 0), (15, 37): (15, 0), (15, 38): (15, 0), (15, 39): (15, 0), (15, 40): (15, 0),  # noqa: B950
    (16, 16): (16, 23), (16, 17): (16, 23), (16, 18): (16, 22), (16, 19): (16, 21), (16, 20): (16, 20), (16, 21): (16, 19), (16, 22): (16, 18), (16, 23): (16, 17), (16, 24): (16, 16), (16, 25): (16, 15), (16, 26): (16, 14), (16, 27): (16, 13), (16, 28): (16, 12), (16, 29): (16, 11), (16, 30): (16, 5), (16, 31): (16, 5), (16, 32): (16, 5), (16, 33): (16, 5), (16, 34): (16, 5), (16, 35): (16, 5), (16, 36): (16, 0), (16, 37): (16, 0), (16, 38): (16, 0), (16, 39): (16, 0), (16, 40): (16, 0),  # noqa: B950
    (17, 17): (16, 23), (17, 18): (16, 22), (17, 19): (16, 21), (17, 20): (16, 20), (17, 21): (16, 19), (17, 22): (16, 18), (17, 23): (16, 17), (17, 24): (16, 16), (17, 25): (16, 15), (17, 26): (16, 14), (17, 27): (16, 13), (17, 28): (16, 12), (17, 29): (16, 11), (17, 30): (16, 5), (17, 31): (16, 5), (17, 32): (16, 5), (17, 33): (16, 5), (17, 34): (16, 5), (17, 35): (16, 5), (17, 36): (16, 0), (17, 37): (16, 0), (17, 38): (16, 0), (17, 39): (16, 0), (17, 40): (16, 0),  # noqa: B950
    (18, 18): (18, 22), (18, 19): (18, 21), (18, 20): (18, 20), (18, 21): (18, 19), (18, 22): (18, 18), (18, 23): (18, 17), (18, 24): (18, 16), (18, 25): (18, 15), (18, 26): (18, 14), (18, 27): (18, 13), (18, 28): (18, 12), (18, 29): (18, 11), (18, 30): (18, 5), (18, 31): (18, 5), (18, 32): (18, 5), (18, 33): (18, 5), (18, 34): (18, 5), (18, 35): (18, 5), (18, 36): (18, 0), (18, 37): (18, 0), (18, 38): (18, 0), (18, 39): (18, 0), (18, 40): (18, 0),  # noqa: B950
    (19, 19): (19, 21), (19, 20): (19, 20), (19, 21): (19, 19), (19, 22): (19, 18), (19, 23): (19, 17), (19, 24): (19, 16), (19, 25): (19, 15), (19, 26): (19, 14), (19, 27): (19, 13), (19, 28): (19, 12), (19, 29): (19, 11), (19, 30): (19, 5), (19, 31): (19, 5), (19, 32): (19, 5), (19, 33): (19, 5), (19, 34): (19, 5), (19, 35): (19, 5), (19, 36): (19, 0), (19, 37): (19, 0), (19, 38): (19, 0), (19, 39): (19, 0), (19, 40): (19, 0),  # noqa: B950
    (20, 20): (20, 20), (20, 21): (20, 19), (20, 22): (20, 18), (20, 23): (20, 17), (20, 24): (20, 16), (20, 25): (20, 15), (20, 26): (20, 14), (20, 27): (20, 13), (20, 28): (20, 12), (20, 29): (20, 11), (20, 30): (20, 5), (20, 31): (20, 5), (20, 32): (20, 5), (20, 33): (20, 5), (20, 34): (20, 5), (20, 35): (20, 5), (20, 36): (20, 0), (20, 37): (20, 0), (20, 38): (20, 0), (20, 39): (20, 0), (20, 40): (20, 0),  # noqa: B950
    (21, 21): (21, 19), (21, 22): (21, 18), (21, 23): (21, 17), (21, 24): (21, 16), (21, 25): (21, 15), (21, 26): (21, 14), (21, 27): (21, 13), (21, 28): (21, 12), (21, 29): (21, 11), (21, 30): (21, 5), (21, 31): (21, 5), (21, 32): (21, 5), (21, 33): (21, 5), (21, 34): (21, 5), (21, 35): (21, 5), (21, 36): (21, 0), (21, 37): (21, 0), (21, 38): (21, 0), (21, 39): (21, 0), (21, 40): (21, 0),  # noqa: B950
    (22, 22): (22, 18), (22, 23): (22, 17), (22, 24): (22, 16), (22, 25): (22, 15), (22, 26): (22, 14), (22, 27): (22, 13), (22, 28): (22, 12), (22, 29): (22, 11), (22, 30): (22, 5), (22, 31): (22, 5), (22, 32): (22, 5), (22, 33): (22, 5), (22, 34): (22, 5), (22, 35): (22, 5), (22, 36): (22, 0), (22, 37): (22, 0), (22, 38): (22, 0), (22, 39): (22, 0), (22, 40): (22, 0),  # noqa: B950
    (23, 23): (23, 17), (23, 24): (23, 16), (23, 25): (23, 15), (23, 26): (23, 14), (23, 27): (23, 13), (23, 28): (23, 12), (23, 29): (23, 11), (23, 30): (23, 5), (23, 31): (23, 5), (23, 32): (23, 5), (23, 33): (23, 5), (23, 34): (23, 5), (23, 35): (23, 5), (23, 36): (23, 0), (23, 37): (23, 0), (23, 38): (23, 0), (23, 39): (23, 0), (23, 40): (23, 0),  # noqa: B950
    (24, 24): (24, 16), (24, 25): (24, 15), (24, 26): (24, 14), (24, 27): (24, 13), (24, 28): (24, 12), (24, 29): (24, 11), (24, 30): (24, 5), (24, 31): (24, 5), (24, 32): (24, 5), (24, 33): (24, 5), (24, 34): (24, 5), (24, 35): (24, 5), (24, 36): (24, 0), (24, 37): (24, 0), (24, 38): (24, 0), (24, 39): (24, 0), (24, 40): (24, 0),  # noqa: B950
    (25, 25): (25, 15), (25, 26): (25, 14), (25, 27): (25, 13), (25, 28): (25, 12), (25, 29): (25, 11), (25, 30): (25, 5), (25, 31): (25, 5), (25, 32): (25, 5), (25, 33): (25, 5), (25, 34): (25, 5), (25, 35): (25, 5), (25, 36): (25, 0), (25, 37): (25, 0), (25, 38): (25, 0), (25, 39): (25, 0), (25, 40): (25, 0),  # noqa: B950
    (26, 26): (26, 14), (26, 27): (26, 13), (26, 28): (26, 12), (26, 29): (26, 11), (26, 30): (26, 5), (26, 31): (26, 5), (26, 32): (26, 5), (26, 33): (26, 5), (26, 34): (26, 5), (26, 35): (26, 5), (26, 36): (26, 0), (26, 37): (26, 0), (26, 38): (26, 0), (26, 39): (26, 0), (26, 40): (26, 0),  # noqa: B950
    (27, 27): (27, 13), (27, 28): (27, 12), (27, 29): (27, 11), (27, 30): (27, 5), (27, 31): (27, 5), (27, 32): (27, 5), (27, 33): (27, 5), (27, 34): (27, 5), (27, 35): (27, 5), (27, 36): (27, 0), (27, 37): (27, 0), (27, 38): (27, 0), (27, 39): (27, 0), (27, 40): (27, 0),  # noqa: B950
    (28, 28): (28, 12), (28, 29): (28, 11), (28, 30): (28, 5), (28, 31): (28, 5), (28, 32): (28, 5), (28, 33): (28, 5), (28, 34): (28, 5), (28, 35): (28, 5), (28, 36): (28, 0), (28, 37): (28, 0), (28, 38): (28, 0), (28, 39): (28, 0), (28, 40): (28, 0),  # noqa: B950
    (29, 29): (29, 11), (29, 30): (29, 5), (29, 31): (29, 5), (29, 32): (29, 5), (29, 33): (29, 5), (29, 34): (29, 5), (29, 35): (29, 5), (29, 36): (29, 0), (29, 37): (29, 0), (29, 38): (29, 0), (29, 39): (29, 0), (29, 40): (29, 0),  # noqa: B950
    (30, 30): (29, 5), (30, 31): (29, 5), (30, 32): (29, 5), (30, 33): (29, 5), (30, 34): (29, 5), (30, 35): (29, 5), (30, 36): (29, 0), (30, 37): (29, 0), (30, 38): (29, 0), (30, 39): (29, 0), (30, 40): (29, 0),  # noqa: B950
    (31, 31): (29, 5), (31, 32): (29, 5), (31, 33): (29, 5), (31, 34): (29, 5), (31, 35): (29, 5), (31, 36): (29, 0), (31, 37): (29, 0), (31, 38): (29, 0), (31, 39): (29, 0), (31, 40): (29, 0),  # noqa: B950
    (32, 32): (29, 5), (32, 33): (29, 5), (32, 34): (29, 5), (32, 35): (29, 5), (32, 36): (29, 0), (32, 37): (29, 0), (32, 38): (29, 0), (32, 39): (29, 0), (32, 40): (29, 0),  # noqa: B950
    (33, 33): (29, 5), (33, 34): (29, 5), (33, 35): (29, 5), (33, 36): (29, 0), (33, 37): (29, 0), (33, 38): (29, 0), (33, 39): (29, 0), (33, 40): (29, 0),  # noqa: B950
    (34, 34): (29, 5), (34, 35): (29, 5), (34, 36): (29, 0), (34, 37): (29, 0), (34, 38): (29, 0), (34, 39): (29, 0), (34, 40): (29, 0),  # noqa: B950
    (35, 35): (35, 5), (35, 36): (35, 0), (35, 37): (35, 0), (35, 38): (35, 0), (35, 39): (35, 0), (35, 40): (35, 0),  # noqa: B950
    (36, 36): (36, 0), (36, 37): (36, 0), (36, 38): (36, 0), (36, 39): (36, 0), (36, 40): (36, 0),  # noqa: B950
    (37, 37): (36, 0), (37, 38): (36, 0), (37, 39): (36, 0), (37, 40): (36, 0),
    (38, 38): (36, 0), (38, 39): (36, 0), (38, 40): (36, 0),
    (39, 39): (36, 0), (39, 40): (36, 0),
    (40, 40): (36, 0),
}
# fmt:on


def test_lines() -> None:
    def check_line_formatting(source: str, expected: str, mode: Mode) -> None:
        line_start = mode.line_range[0] - 1 if mode.line_range else 1
        line_end = mode.line_range[1] + 1 if mode.line_range else 1

        formatted = format_str(source, mode=mode)

        splitted_source = source.split("\n")
        splitted_formatted = formatted.split("\n")

        assert (
            "\n".join(
                splitted_source[:line_start]
                + splitted_formatted[line_start:-line_end]
                + splitted_source[-line_end:]
            )
            == formatted
        )
        formatted_span = "\n".join(splitted_formatted[line_start:-line_end])
        assert formatted_span in expected

    args, source, _ = read_data_with_mode("miscellaneous", "lines.py", data=True)
    mode = args.mode
    # We do this only once to speed up the tests
    source_node = lib2to3_parse(source, mode.target_versions)
    expected = format_str(source, mode=mode)

    lines = source.count("\n") + 1

    for line_start in range(1, lines):
        for line_end in range(line_start, lines):
            line_range = calculate_line_range(
                (line_start, line_end), source, mode, source_node
            )
            expected_line_range = expected_line_ranges[(line_start, line_end)]
            assert line_range == expected_line_range

    # Test only all ranges [1,x] + [x,EOF] to not slow down the tests too much
    ranges = [(1, line_end) for line_end in range(1, lines)] + [
        (line_start, lines - 1) for line_start in range(1, lines)
    ]
    for line_start, line_end in ranges:
        check_line_formatting(
            source,
            expected,
            replace(
                mode,
                # We validated the ranges already above, so let's use the expected ones
                line_range=expected_line_ranges[(line_start, line_end)],
            ),
        )

    # Uncomment to run all permutations – SLOW
    """
    for line_start in range(1, lines):
        for line_end in range(line_start, lines):
            check_line_formatting(
                source,
                expected,
                replace(
                    mode,
                    line_range=expected_line_ranges[(line_start, line_end)],
                ),
            )
    """
