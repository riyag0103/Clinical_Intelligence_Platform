import cv2
import numpy as np

# ----------------------------------
# REMOVE SMALL NOISE BLOBS
# ----------------------------------
def remove_small_components(
    mask,
    min_size=500
):

    # ----------------------------------
    # CONNECTED COMPONENTS
    # ----------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # ----------------------------------
    # OUTPUT MASK
    # ----------------------------------
    cleaned_mask = np.zeros_like(mask)

    # ----------------------------------
    # KEEP LARGE COMPONENTS
    # ----------------------------------
    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_size:

            cleaned_mask[
                labels == i
            ] = 1

    return cleaned_mask

# ----------------------------------
# MORPHOLOGICAL CLEANUP
# ----------------------------------
def morphological_cleanup(mask):

    kernel = np.ones((5, 5), np.uint8)

    # ----------------------------------
    # OPENING
    # removes tiny noise
    # ----------------------------------
    cleaned = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel
    )

    # ----------------------------------
    # CLOSING
    # fills small holes
    # ----------------------------------
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    return cleaned

# ----------------------------------
# COMPLETE POSTPROCESSING
# ----------------------------------
def postprocess_mask(mask):

    # ----------------------------------
    # REMOVE SMALL COMPONENTS
    # ----------------------------------
    cleaned = remove_small_components(
        mask,
        min_size=500
    )

    # ----------------------------------
    # MORPHOLOGICAL CLEANUP
    # ----------------------------------
    cleaned = morphological_cleanup(
        cleaned
    )

    return cleaned