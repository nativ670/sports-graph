import argparse
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
RAW_DIR = Path("data/raw/metrica")


def game_files(game: int) -> dict[str, str]:
    """Map local file name -> URL for one sample game (CSV format)."""
    stem = f"Sample_Game_{game}"
    return {
        name: f"{BASE_URL}/{stem}/{name}"
        for name in [
            f"{stem}_RawEventsData.csv",
            f"{stem}_RawTrackingData_Away_Team.csv",
            f"{stem}_RawTrackingData_Home_Team.csv",
        ]
    }


def download(url: str, dest: Path) -> None:
    """Stream url to dest; skip if dest already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Skipping {dest} (already exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    part.replace(dest)


def main() -> None:
    """Parse --game (default 1), download its files, print each file's size."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game", type=int, default=1, choices=[1, 2], help="Sample game number (1-2)"
    )
    args = parser.parse_args()

    for local_name, url in game_files(args.game).items():
        dest = RAW_DIR / local_name
        download(url, dest)
        print(f"{dest}: {dest.stat().st_size} bytes")


if __name__ == "__main__":
    main()
