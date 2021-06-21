"""
Defines preprocessing functions for the segmentation gan.
"""

import numpy as np


def coshuffle_arrays(arr1, arr2):
    """
    Shuffles two arrays in unison along their first axis.
    For example, coshuffle_arrays([0, 1, 2], [3, 4, 5]) can return
    [0, 2, 1], [3, 5, 4] but not [0, 2, 1], [5, 4, 3].
    The arrays' first axes must be of the dimension.
    """
    assert arr1.shape[0] == arr2.shape[
        0], "arr1 and arr2's first axes should be of same dimension"

    perm = np.random.permutation(arr1.shape[0])
    return arr1[perm], arr2[perm]


def coshuffle_lists(list1, list2):
    """
    Shuffles two lists in unison. Both lists must have
    the same length.
    """
    assert len(list1) == len(
        list2), "list1 and list2 must have the same length"

    perm = np.random.permutation(len(list1))
    return [list1[i] for i in perm], [list2[i] for i in perm]
