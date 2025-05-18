import rawpy
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import typer
from pathlib import Path


def load_cr3(path: Path) -> np.ndarray:
    """Read a Canon CR3 RAW and return an 8-bit RGB NumPy array."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()
    return rgb


def process_files(a: Path, b: Path, c: Path, output_dir: Path, threshold: int = 165) -> None:
    a_rgb = load_cr3(a)
    b_rgb = load_cr3(b)
    c_rgb = load_cr3(c)

    # pick green channel
    a_gray = a_rgb[:, :, 1]
    b_gray = b_rgb[:, :, 1]
    c_gray = c_rgb[:, :, 1]

    # compute SSIM
    score_ab, diff_ab = compare_ssim(a_gray, b_gray, full=True)
    score_bc, diff_bc = compare_ssim(b_gray, c_gray, full=True)

    # scale diffs to 0-255
    diff_ab = (diff_ab * 255).astype('uint8')
    diff_bc = (diff_bc * 255).astype('uint8')

    print(f"SSIM a vs b: {score_ab:.4f}")
    print(f"SSIM b vs c: {score_bc:.4f}")

    # threshold and invert
    _, thresh_ab = cv2.threshold(diff_ab, threshold, 255, cv2.THRESH_BINARY_INV)
    _, thresh_bc = cv2.threshold(diff_bc, threshold, 255, cv2.THRESH_BINARY_INV)

    # save outputs to the working folder
    output_ab = output_dir / 'diff_ab.jpg'
    output_bc = output_dir / 'diff_bc.jpg'
    
    cv2.imwrite(str(output_ab), thresh_ab)
    cv2.imwrite(str(output_bc), thresh_bc)
    print(f"Saved {output_ab} and {output_bc}")

app = typer.Typer(help="Compute SSIM diffs between CR3 files.")

@app.command()
def main(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Folder containing CR3 files"),
    a: Path = typer.Option(None, "--a", exists=True, file_okay=True, dir_okay=False, help="First CR3 file"),
    b: Path = typer.Option(None, "--b", exists=True, file_okay=True, dir_okay=False, help="Second CR3 file"),
    c: Path = typer.Option(None, "--c", exists=True, file_okay=True, dir_okay=False, help="Third CR3 file")
):
    """Process the last three CR3 files in a folder or specified CR3 files."""
    if a or b or c:
        if not all([a, b, c]):
            typer.echo("Error: --a, --b, and --c must all be provided", err=True)
            raise typer.Exit(code=1)
        a_file, b_file, c_file = a, b, c
        # Use the folder of the first file as output directory
        output_dir = a_file.parent
    else:
        cr3_files = sorted(folder.glob("*.CR3"), key=lambda x: x.stat().st_mtime)
        if len(cr3_files) < 3:
            typer.echo("Error: Not enough CR3 files in folder", err=True)
            raise typer.Exit(code=1)
        a_file, b_file, c_file = cr3_files[-3:]
        output_dir = folder

    process_files(a_file, b_file, c_file, output_dir)

if __name__ == "__main__":
    app() 