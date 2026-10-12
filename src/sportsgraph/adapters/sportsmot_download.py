import time
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Hellfire98/sportsmot"
REPO_TYPE = "dataset"

SPLIT = "test"
SEQUENCE = "v_1UDUODIBSsc_c103"

OUTPUT_DIR = Path("data/raw/sportsmot")

# Set to None to download the entire sequence.
# For development, you can use e.g. 100.
MAX_FRAMES = 100

# Number of attempts for a failed HTTP request.
MAX_RETRIES = 3

# Seconds to wait between retries.
RETRY_DELAY = 2

# HTTP timeout.
TIMEOUT = 60


# ============================================================
# PATHS
# ============================================================

REMOTE_SEQUENCE_DIR = f"{SPLIT}/{SEQUENCE}"

LOCAL_SEQUENCE_DIR = OUTPUT_DIR / SPLIT / SEQUENCE
LOCAL_IMG_DIR = LOCAL_SEQUENCE_DIR / "img1"
LOCAL_GT_DIR = LOCAL_SEQUENCE_DIR / "gt"


# ============================================================
# HELPERS
# ============================================================


def get_repository_files():
    """Get the file list from the Hugging Face repository."""

    print("Getting repository file list...")

    api = HfApi()

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    print(f"Found {len(files)} files in repository.")

    return files


def get_sequence_files(files):
    """Return all files belonging to the selected sequence."""

    prefix = f"{REMOTE_SEQUENCE_DIR}/"

    sequence_files = [file for file in files if file.startswith(prefix)]

    return sorted(sequence_files)


def get_frame_files(sequence_files):
    """Return only JPG frame files from img1."""

    img_prefix = f"{REMOTE_SEQUENCE_DIR}/img1/"

    frames = [
        file
        for file in sequence_files
        if file.startswith(img_prefix) and file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Sort numerically by filename.
    frames.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x)

    return frames


def get_metadata_files(sequence_files):
    """Return sequence metadata files."""

    metadata = []

    for file in sequence_files:
        # Sequence information
        if file == f"{REMOTE_SEQUENCE_DIR}/seqinfo.ini":
            metadata.append(file)

        # Ground truth
        elif file.startswith(f"{REMOTE_SEQUENCE_DIR}/gt/"):
            metadata.append(file)

    return metadata


def download_file(remote_path, local_path):
    """
    Download a single file directly over HTTP.

    This intentionally does NOT use hf_hub_download(),
    avoiding the Xet reconstruction problem seen with 000002.jpg.
    """

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip files that are already downloaded.
    if local_path.exists() and local_path.stat().st_size > 0:
        return True

    # Hugging Face resolve endpoint.
    encoded_path = quote(remote_path, safe="/")

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{encoded_path}?download=true"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=TIMEOUT,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                # Write to a temporary file first.
                # This prevents a partially downloaded file from
                # being mistaken for a completed frame.
                temp_path = local_path.with_suffix(local_path.suffix + ".part")

                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                # Move completed file into final location.
                temp_path.replace(local_path)

                return True

        except Exception as e:
            print(f"\nFailed: {remote_path}\nAttempt {attempt}/{MAX_RETRIES}\nError: {e}")

            # Remove incomplete download.
            temp_path = local_path.with_suffix(local_path.suffix + ".part")

            if temp_path.exists():
                temp_path.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return False


# ============================================================
# MAIN
# ============================================================


def download_sequence():

    print()
    print("=" * 70)
    print("SportsMOT sequence downloader")
    print("=" * 70)
    print()

    print(f"Repository : {REPO_ID}")
    print(f"Split      : {SPLIT}")
    print(f"Sequence   : {SEQUENCE}")
    print(f"Output     : {LOCAL_SEQUENCE_DIR}")
    print()

    # --------------------------------------------------------
    # Get repository file list
    # --------------------------------------------------------

    files = get_repository_files()

    sequence_files = get_sequence_files(files)

    if not sequence_files:
        raise RuntimeError(f"No files found for sequence: {REMOTE_SEQUENCE_DIR}")

    print(f"Found {len(sequence_files)} files belonging to the sequence.")

    # --------------------------------------------------------
    # Find frames
    # --------------------------------------------------------

    frames = get_frame_files(sequence_files)

    if not frames:
        raise RuntimeError(f"No image frames found in {REMOTE_SEQUENCE_DIR}/img1/")

    print(f"Found {len(frames)} image frames.")

    # --------------------------------------------------------
    # Limit frames for development
    # --------------------------------------------------------

    if MAX_FRAMES is not None:
        frames_to_download = frames[:MAX_FRAMES]

        print(f"Development mode: downloading first {len(frames_to_download)} frames.")

    else:
        frames_to_download = frames

        print("Downloading all frames.")

    print()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    LOCAL_IMG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download frames
    # --------------------------------------------------------

    successful = 0
    failed = []

    print("Downloading frames...")

    for remote_path in tqdm(
        frames_to_download,
        desc="Frames",
        unit="frame",
    ):
        filename = Path(remote_path).name

        local_path = LOCAL_IMG_DIR / filename

        success = download_file(
            remote_path,
            local_path,
        )

        if success:
            successful += 1

        else:
            failed.append(remote_path)

    # --------------------------------------------------------
    # Download metadata
    # --------------------------------------------------------

    metadata_files = get_metadata_files(sequence_files)

    print()
    print("Downloading metadata...")

    for remote_path in metadata_files:
        relative_path = Path(remote_path).relative_to(REMOTE_SEQUENCE_DIR)

        local_path = LOCAL_SEQUENCE_DIR / relative_path

        success = download_file(
            remote_path,
            local_path,
        )

        if not success:
            failed.append(remote_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"Frames downloaded : {successful}/{len(frames_to_download)}")
    print(f"Failed files      : {len(failed)}")
    print(f"Output directory  : {LOCAL_SEQUENCE_DIR}")
    print()

    if failed:
        print("Failed files:")

        for file in failed:
            print(f"  - {file}")

        print()

    else:
        print("Everything downloaded successfully.")

    print()
    print("Sequence:")
    print(SEQUENCE)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    download_sequence()
