from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MUL1_MULTIPLIER = 0x83DCD12D
K2_STATE_BITS = 16
K2_BRANCH_BITS = 2
K2_TILE_VALUES = 256
K2_EDGE_COUNT = 1 << (K2_STATE_BITS - K2_BRANCH_BITS)

# The 111 unreachable byte-sum slots are retained as declared aliases so the
# physical representation remains one native FP16[1024] LUT. Runtime decode
# computes a byte sum directly and never consults a state map.
K2_CHILD_PARENT_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 510), (2, 502), (3, 511), (4, 516), (5, 519), (6, 499), (7, 532),
    (8, 491), (9, 512), (10, 517), (11, 538), (12, 477), (13, 484),
    (14, 501), (15, 507), (16, 515), (17, 518), (18, 485), (19, 503),
    (20, 528), (21, 474), (22, 494), (23, 495), (24, 497), (25, 505),
    (26, 524), (27, 525), (28, 493), (29, 498), (30, 523), (31, 527),
    (33, 530), (34, 496), (35, 504), (36, 531), (37, 465), (38, 535),
    (39, 513), (40, 486), (41, 509), (42, 537), (43, 492), (44, 514),
    (45, 521), (46, 529), (47, 534), (49, 536), (52, 541), (54, 489),
    (55, 471), (57, 480), (58, 522), (59, 483), (60, 488), (63, 508),
    (65, 520), (71, 544), (73, 479), (944, 487), (947, 478), (958, 490),
    (960, 506), (961, 475), (963, 481), (966, 547), (974, 473),
    (975, 459), (976, 500), (977, 545), (978, 526), (979, 546),
    (980, 467), (981, 539), (982, 540), (984, 542), (985, 533),
    (987, 554), (988, 460), (990, 468), (991, 562), (992, 458),
    (993, 563), (994, 476), (995, 472), (996, 556), (997, 482),
    (998, 553), (999, 457), (1000, 469), (1001, 549), (1002, 551),
    (1003, 543), (1004, 550), (1006, 463), (1007, 552), (1008, 462),
    (1009, 470), (1010, 555), (1011, 464), (1012, 466), (1013, 557),
    (1014, 456), (1015, 558), (1016, 564), (1017, 444), (1018, 548),
    (1019, 451), (1020, 452), (1021, 559), (1022, 561), (1023, 570),
)


def k2_parent_lut_fp16() -> np.ndarray:
    """Return the 1024 exact FP16 parent values of the mul1 codebook."""

    half_inputs = (np.arange(1024, dtype=np.uint16) + np.uint16(0x6400)).view(
        np.float16
    )
    inverse = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0]
    bias = np.array([0xC931], dtype=np.uint16).view(np.float16)[0]
    # A half FMA has one final rounding. Float64 evaluates the exact half inputs
    # and the final cast performs that single rounding.
    return (
        half_inputs.astype(np.float64) * float(inverse) + float(bias)
    ).astype(np.float16)


def k2_lut_fp16() -> np.ndarray:
    """Return the native FP16[1024] LUT with 913 parents and 111 aliases."""

    lut = k2_parent_lut_fp16()
    for child, parent in K2_CHILD_PARENT_PAIRS:
        lut[child] = lut[parent]
    return lut


def mul1_lut_indices(states: np.ndarray | Sequence[int]) -> np.ndarray:
    """Map uint16 trellis states to native LUT slots without a state table."""

    state = np.asarray(states, dtype=np.uint16).astype(np.uint64)
    product = np.bitwise_and(
        state * np.uint64(MUL1_MULTIPLIER), np.uint64(0xFFFFFFFF)
    )
    return (
        product % 256
        + (product // 256) % 256
        + (product // 65536) % 256
        + (product // 16777216) % 256
    ).astype(np.uint16)


def decode_k2_states(
    states: np.ndarray | Sequence[int], lut: np.ndarray | None = None
) -> np.ndarray:
    """Decode uint16 trellis states through the native procedural LUT."""

    values = k2_lut_fp16() if lut is None else np.asarray(lut)
    if values.shape != (1024,) or values.dtype != np.float16:
        raise ValueError("K2 LUT must have shape (1024,) and dtype float16")
    return values[mul1_lut_indices(states)]


def tensor_core_permutation() -> np.ndarray:
    """Return the 16x16 EXL3 trellis element order."""

    permutation = np.empty(256, dtype=np.int32)
    for thread in range(32):
        row0 = (thread % 4) * 2
        row1 = row0 + 1
        row2 = row0 + 8
        row3 = row0 + 9
        column0 = thread // 4
        column1 = column0 + 8
        permutation[thread * 8 : thread * 8 + 8] = (
            row0 * 16 + column0,
            row1 * 16 + column0,
            row2 * 16 + column0,
            row3 * 16 + column0,
            row0 * 16 + column1,
            row1 * 16 + column1,
            row2 * 16 + column1,
            row3 * 16 + column1,
        )
    return permutation


def cyclic_states_from_codes(codes: np.ndarray) -> np.ndarray:
    """Expand packed two-bit branches into their cyclic uint16 states."""

    branches = np.asarray(codes)
    if branches.shape[-1] != K2_TILE_VALUES:
        raise ValueError("K2 branch stream must end in 256 values")
    branches = np.bitwise_and(branches.astype(np.uint16), 3)
    state = np.zeros(branches.shape[:-1], dtype=np.uint16)
    for index in range(K2_TILE_VALUES):
        state = np.bitwise_and(
            np.left_shift(state, K2_BRANCH_BITS) | branches[..., index], 0xFFFF
        )
    states = np.empty(branches.shape, dtype=np.uint16)
    for index in range(K2_TILE_VALUES):
        state = np.bitwise_and(
            np.left_shift(state, K2_BRANCH_BITS) | branches[..., index], 0xFFFF
        )
        states[..., index] = state
    return states


def pack_k2_trellis(encoded: np.ndarray) -> np.ndarray:
    """Pack uint16 states to the canonical 32-word K2 tile wire layout."""

    states = np.asarray(encoded)
    if states.shape[-1] != K2_TILE_VALUES:
        raise ValueError("encoded K2 trellis must end in 256 states")
    branches = np.bitwise_and(states.astype(np.uint16), 3).reshape(
        *states.shape[:-1], 16, 16
    )
    words: list[np.ndarray] = []
    for values in (branches[..., :8], branches[..., 8:]):
        word = np.zeros(values.shape[:-1], dtype=np.uint16)
        for index in range(8):
            word = np.bitwise_or(
                word,
                np.left_shift(values[..., index], np.uint16(14 - 2 * index)),
            )
        words.append(word)
    # Canonical wire order swaps each adjacent uint16 pair.
    return np.stack((words[1], words[0]), axis=-1).reshape(
        *states.shape[:-1], 32
    )


def unpack_k2_trellis(packed: np.ndarray) -> np.ndarray:
    """Unpack canonical K2 words to the parent-equivalent cyclic state path."""

    words = np.asarray(packed)
    if words.shape[-1] != 32:
        raise ValueError("packed K2 trellis must end in 32 uint16 words")
    pairs = words.astype(np.uint16).reshape(*words.shape[:-1], 16, 2)
    branches = np.empty((*words.shape[:-1], 16, 16), dtype=np.uint16)
    for half, source in ((0, pairs[..., 1]), (1, pairs[..., 0])):
        for index in range(8):
            branches[..., half * 8 + index] = np.bitwise_and(
                np.right_shift(source, np.uint16(14 - 2 * index)), 3
            )
    return cyclic_states_from_codes(branches.reshape(*words.shape[:-1], 256))
