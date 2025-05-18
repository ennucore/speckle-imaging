import rawpy
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def load_cr3(path):
    """Read a Canon CR3 RAW and return an 8-bit RGB NumPy array."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb  # (H, W, 3), uint8

# ————— Load all three raws —————
a_rgb = load_cr3('a.CR3')
b_rgb = load_cr3('b.CR3')
c_rgb = load_cr3('c.CR3')

# ————— Pick green channel (or convert to gray) —————
a_gray = a_rgb[:, :, 1]
b_gray = b_rgb[:, :, 1]
c_gray = c_rgb[:, :, 1]

# ————— Compute SSIM diffs —————
score_ab, diff_ab = compare_ssim(a_gray, b_gray, full=True)
score_bc, diff_bc = compare_ssim(b_gray, c_gray, full=True)

# scale to 0–255
diff_ab = (diff_ab * 255).astype('uint8')
diff_bc = (diff_bc * 255).astype('uint8')

print(f"SSIM a vs b: {score_ab:.4f}")
print(f"SSIM b vs c: {score_bc:.4f}")

# ————— Fixed threshold param —————
fixed_thresh_value = 165

# ————— Threshold and save —————
_, thresh_ab = cv2.threshold(
    diff_ab,
    fixed_thresh_value,
    255,
    cv2.THRESH_BINARY_INV
)
_, thresh_bc = cv2.threshold(
    diff_bc,
    fixed_thresh_value,
    255,
    cv2.THRESH_BINARY_INV
)

cv2.imwrite('out-c.jpg', thresh_ab)  # a vs b
cv2.imwrite('out.jpg',   thresh_bc)  # b vs c