"""Rasterize the intuition diagrams in this directory to docs/src/assets/intuition.

The `.html` files here are the editable sources of the figures on the
"How SPlit works" page; the docs embed only the exported PNGs. Re-run this
after editing a source:

    uv run --with playwright python docs/diagrams/export.py [chromium-binary]

The optional argument points at a Chromium binary when the Playwright
browsers are not installed (`playwright install chromium` installs them).
Text is set in Pretendard (install it locally; Helvetica is the fallback)
and math in STIX Two Text, fetched from Google Fonts at render time.
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / 'src' / 'assets' / 'intuition'
SCALE = 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    launch = {'executable_path': sys.argv[1]} if len(sys.argv) > 1 else {}
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        page = browser.new_page(device_scale_factor=SCALE)
        for src in sorted(HERE.glob('*.html')):
            page.goto(f'file://{src}')
            page.wait_for_load_state('networkidle')
            out = OUT / f'{src.stem}.png'
            page.locator('svg').first.screenshot(path=out, omit_background=True)
            print(f'{src.name} -> {out.relative_to(HERE.parent.parent)}')
        browser.close()


if __name__ == '__main__':
    main()
