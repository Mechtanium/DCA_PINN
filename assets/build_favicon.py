"""Render the SVG sources in this folder into ../favicon.ico.

The icon PERD shows on the desktop and in the Store is the repository's
``favicon.ico``: an ICO or PNG file of at most 256 KiB. This script renders
``favicon.svg`` at 256, 128, 64 and 48 px, and the simplified drawings
``favicon-32.svg`` and ``favicon-16.svg`` at their own sizes, then packs the
six PNG images into one ICO::

    pip install cairosvg
    python assets/build_favicon.py

cairosvg needs the system cairo library (``libcairo2`` on Debian and Ubuntu).
"""

from __future__ import annotations

import struct
from pathlib import Path

import cairosvg

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "favicon.ico"

#: Pixel size and source drawing of the ICO's images.
IMAGES = (
    (256, "favicon.svg"),
    (128, "favicon.svg"),
    (64, "favicon.svg"),
    (48, "favicon.svg"),
    (32, "favicon-32.svg"),
    (16, "favicon-16.svg"),
)

#: PERD's limit for a workflow icon.
MAX_BYTES = 256 * 1024


def render(source: str, size: int) -> bytes:
    """The drawing as a PNG of ``size`` x ``size`` pixels."""
    return cairosvg.svg2png(
        url=str(HERE / source), output_width=size, output_height=size
    )


def pack_ico(images: list[tuple[int, bytes]]) -> bytes:
    """An ICO file holding PNG images (the PNG-in-ICO form, Windows Vista on).

    The directory stores widths and heights in one byte, so 256 is written as 0.
    """
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    directory = b""
    data = b""
    for size, png in images:
        directory += struct.pack(
            "<BBBBHHII",
            size % 256,  # width
            size % 256,  # height
            0,  # palette colours
            0,  # reserved
            1,  # colour planes
            32,  # bits per pixel
            len(png),
            offset + len(data),
        )
        data += png
    return header + directory + data


def main() -> None:
    images = [(size, render(source, size)) for size, source in sorted(IMAGES)]
    ico = pack_ico(images)
    if len(ico) > MAX_BYTES:
        raise SystemExit(f"favicon.ico is {len(ico)} bytes; PERD's limit is {MAX_BYTES}")
    OUT.write_bytes(ico)
    sizes = ", ".join(str(size) for size, _ in images)
    print(f"wrote {OUT.name}: {len(ico)} bytes, PNG images at {sizes} px")


if __name__ == "__main__":
    main()
