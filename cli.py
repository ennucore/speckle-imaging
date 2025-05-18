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


def load_jpeg(path: Path) -> np.ndarray:
    """Read a JPEG image and return an RGB NumPy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image(path: Path) -> np.ndarray:
    """Load an image file based on its extension."""
    ext = path.suffix.lower()
    if ext in ['.cr3', '.cr2', '.crw']:
        return load_cr3(path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return load_jpeg(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def process_files(a: Path, b: Path, c: Path, output_dir: Path, threshold: int = 150, 
                   use_otsu: bool = False, blur_kernel: int = 15) -> None:
    a_rgb = load_image(a)
    b_rgb = load_image(b)
    c_rgb = load_image(c)

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

    # Apply gaussian blur
    blur_ab = cv2.GaussianBlur(diff_ab, (blur_kernel, blur_kernel), 0)
    blur_bc = cv2.GaussianBlur(diff_bc, (blur_kernel, blur_kernel), 0)

    # Threshold and invert
    if use_otsu:
        _, thresh_ab = cv2.threshold(diff_ab, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        _, thresh_bc = cv2.threshold(diff_bc, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    else:
        _, thresh_ab = cv2.threshold(diff_ab, threshold, 255, cv2.THRESH_BINARY_INV)
        _, thresh_bc = cv2.threshold(diff_bc, threshold, 255, cv2.THRESH_BINARY_INV)

    # save outputs to the working folder
    output_ab = output_dir / 'diff_ab.jpg'
    output_bc = output_dir / 'diff_bc.jpg'
    
    # Save blurred images
    output_blur_ab = output_dir / 'diff_blur_ab.jpg'
    output_blur_bc = output_dir / 'diff_blur_bc.jpg'
    
    cv2.imwrite(str(output_ab), thresh_ab)
    cv2.imwrite(str(output_bc), thresh_bc)
    cv2.imwrite(str(output_blur_ab), blur_ab)
    cv2.imwrite(str(output_blur_bc), blur_bc)
    
    print(f"Saved outputs to {output_dir}")

app = typer.Typer(help="Compute SSIM diffs between CR3 files.")

@app.command()
def main(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Folder containing image files"),
    a: Path = typer.Option(None, "--a", help="First image file"),
    b: Path = typer.Option(None, "--b", help="Second image file"),
    c: Path = typer.Option(None, "--c", help="Third image file"),
    threshold: int = typer.Option(150, "--threshold", "-t", help="Threshold value for binary thresholding"),
    use_otsu: bool = typer.Option(False, "--otsu", "-o", help="Use Otsu thresholding instead of fixed threshold"),
    blur_kernel: int = typer.Option(15, "--blur", "-b", help="Kernel size for Gaussian blur (odd number)")
):
    """Process image files for SSIM comparison."""
    if blur_kernel % 2 == 0:
        typer.echo("Error: Blur kernel size must be an odd number", err=True)
        raise typer.Exit(code=1)
        
    if a or b or c:
        if not all([a, b, c]):
            typer.echo("Error: --a, --b, and --c must all be provided", err=True)
            raise typer.Exit(code=1)
        
        # Verify files exist
        for p in [a, b, c]:
            if not p.exists():
                typer.echo(f"Error: File does not exist: {p}", err=True)
                raise typer.Exit(code=1)
                
        a_file, b_file, c_file = a, b, c
        # Use the folder of the first file as output directory
        output_dir = a_file.parent
    else:
        # Try CR3 files first
        cr3_files = sorted(folder.glob("*.CR3"), key=lambda x: x.stat().st_mtime)
        
        # If not enough CR3 files, look for JPEGs
        if len(cr3_files) < 3:
            typer.echo("Not enough CR3 files, looking for JPEG files...")
            jpeg_files = []
            for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
                jpeg_files.extend(folder.glob(ext))
            jpeg_files = sorted(jpeg_files, key=lambda x: x.stat().st_mtime)
            
            # If we have any CR3 files, add them to the JPEG list and resort
            all_files = cr3_files + jpeg_files
            all_files = sorted(all_files, key=lambda x: x.stat().st_mtime)
            
            if len(all_files) < 3:
                typer.echo("Error: Not enough image files found (minimum 3 required)", err=True)
                raise typer.Exit(code=1)
                
            a_file, b_file, c_file = all_files[-3:]
            typer.echo(f"Using files: {a_file.name}, {b_file.name}, {c_file.name}")
        else:
            a_file, b_file, c_file = cr3_files[-3:]
            
        output_dir = folder

    process_files(a_file, b_file, c_file, output_dir, threshold, use_otsu, blur_kernel)

if __name__ == "__main__":
    app() 