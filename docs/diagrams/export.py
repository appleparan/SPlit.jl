"""Export the intuition diagrams in this directory to docs/src/assets/intuition.

The `.html` files here are the editable sources of the figures on the "How
SPlit works" and Methods pages. Colors live in `theme.css` as CSS custom
properties and the sources reference them only as `var(--token)`, so adding
`data-theme="dark"` to `<html>` previews the dark palette in a browser.

This script renders each source twice, once per theme, and writes
`docs/src/assets/intuition/<name>-<light|dark>.svg`. For each export it
resolves every `var()` presentation attribute to the color the browser
computed, adds `width`/`height` from the viewBox, and inlines the fonts the
figure actually uses (Pretendard for prose, STIX Two Text for math) as
subsetted woff2 data URIs, so the SVG renders identically on machines that
have neither font installed. The docs embed both variants and
`docs/src/assets/themed-figures.css` shows the one matching the active
Documenter theme. Re-run this after editing a source:

    uv run --with playwright --with 'fonttools[woff]' python docs/diagrams/export.py [chromium-binary]

The optional argument points at a Chromium binary when the Playwright
browsers are not installed (`playwright install chromium` installs them).
Downloaded Google Fonts files are cached in `docs/diagrams/.cache/`.
"""

import base64
import io
import pathlib
import re
import sys
import urllib.request

from fontTools import subset
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE.parent / 'src' / 'assets' / 'intuition'
CACHE = HERE / '.cache'

THEMES = ('light', 'dark')

PRETENDARD_DIR = pathlib.Path('/usr/share/fonts/truetype/pretendard')
PRETENDARD_FILES = {
    400: 'Pretendard-Regular.ttf',
    600: 'Pretendard-SemiBold.ttf',
    700: 'Pretendard-Bold.ttf',
}
STIX_CSS_URL = (
    'https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;1,400'
)

# Families the sources name first in their font-family lists.
KNOWN_FAMILIES = {'pretendard': 'Pretendard', 'stix two text': 'STIX Two Text'}

# Resolve every var() presentation attribute inside the SVG to its computed
# color, then hand back the serialized SVG plus the text each font face has to
# cover.
COLLECT_JS = """
() => {
  const svg = document.querySelector('svg');
  const props = ['fill', 'stroke', 'stop-color', 'color'];
  for (const el of svg.querySelectorAll('*')) {
    const cs = getComputedStyle(el);
    for (const prop of props) {
      const value = el.getAttribute(prop);
      if (value && value.includes('var(')) {
        el.setAttribute(prop, cs.getPropertyValue(prop).trim());
      }
    }
  }
  // Text that inherits its face from the page CSS (`body { font-family }`)
  // would lose it in the standalone SVG, so pin the computed face, weight and
  // style as attributes wherever the source did not set them explicitly.
  for (const el of svg.querySelectorAll('text')) {
    const cs = getComputedStyle(el);
    if (!el.hasAttribute('font-family')) el.setAttribute('font-family', cs.fontFamily);
    if (!el.hasAttribute('font-weight') && cs.fontWeight !== '400') {
      el.setAttribute('font-weight', cs.fontWeight);
    }
    if (!el.hasAttribute('font-style') && cs.fontStyle !== 'normal') {
      el.setAttribute('font-style', cs.fontStyle);
    }
  }
  const usage = [];
  const walker = document.createTreeWalker(svg, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const el = node.parentElement;
    if (!el || !el.closest('text')) continue;
    const cs = getComputedStyle(el);
    usage.push({
      family: cs.fontFamily,
      weight: cs.fontWeight,
      style: cs.fontStyle,
      text: node.nodeValue,
    });
  }
  return {markup: svg.outerHTML, viewBox: svg.getAttribute('viewBox'), usage};
}
"""


def stix_sources() -> dict[str, str]:
    """Return the STIX Two Text TTF URL per font-style, from the Google Fonts CSS.

    A plain urllib user agent gets the legacy stylesheet, which serves single
    `format('truetype')` files instead of unicode-range-split woff2 slices.
    """
    request = urllib.request.Request(STIX_CSS_URL, headers={'User-Agent': 'Python-urllib'})
    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed https URL
        css = response.read().decode('utf-8')
    faces = {}
    for block in re.findall(r'@font-face\s*\{(.*?)\}', css, re.S):
        style = re.search(r'font-style:\s*([a-z]+)', block)
        url = re.search(r'url\((https://[^)]+?)\)', block)
        if style and url:
            faces[style.group(1)] = url.group(1)
    missing = {'normal', 'italic'} - faces.keys()
    if missing:
        message = f'Google Fonts CSS did not list STIX Two Text faces: {sorted(missing)}'
        raise RuntimeError(message)
    return faces


def cached_download(url: str) -> pathlib.Path:
    """Download `url` into `.cache/` once and return the local path."""
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / url.rsplit('/', 1)[-1]
    if not target.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Python-urllib'})
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed https URL
            target.write_bytes(response.read())
    return target


def font_file(family: str, weight: int, style: str, stix: dict[str, str]) -> pathlib.Path:
    """Return the source TTF for one face."""
    if family == 'Pretendard':
        return PRETENDARD_DIR / PRETENDARD_FILES[weight]
    return cached_download(stix[style])


def subset_woff2(path: pathlib.Path, chars: set[str]) -> bytes:
    """Subset `path` to `chars` and return it as woff2 bytes."""
    options = subset.Options()
    options.flavor = 'woff2'
    options.notdef_outline = True
    options.drop_tables += ['DSIG']
    font = subset.load_font(str(path), options)
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=''.join(sorted(chars)))
    subsetter.subset(font)
    buffer = io.BytesIO()
    subset.save_font(font, buffer, options)
    font.close()
    return buffer.getvalue()


def face_key(record: dict) -> tuple[str, int, str] | None:
    """Map one text run to the (family, weight, style) face that renders it."""
    first = record['family'].split(',')[0].strip().strip('\'"')
    family = KNOWN_FAMILIES.get(first.lower())
    if family is None:
        return None
    if family == 'Pretendard':
        css_weight = int(record['weight'])
        weight = 400 if css_weight < 600 else (600 if css_weight < 700 else 700)
        # Only upright Pretendard files are embedded; the renderer obliques them.
        return (family, weight, 'normal')
    style = 'italic' if record['style'] == 'italic' else 'normal'
    return (family, 400, style)


def font_style_block(usage: list[dict], stix: dict[str, str]) -> str:
    """Build the `<style>` element embedding every face the SVG uses."""
    needed: dict[tuple[str, int, str], set[str]] = {}
    for record in usage:
        key = face_key(record)
        if key is None:
            continue
        needed.setdefault(key, {' '}).update(record['text'])
    rules = []
    for (family, weight, style), chars in sorted(needed.items()):
        data = subset_woff2(font_file(family, weight, style, stix), chars)
        encoded = base64.b64encode(data).decode('ascii')
        rules.append(
            f"@font-face{{font-family:'{family}';font-style:{style};"
            f'font-weight:{weight};font-display:block;'
            f"src:url(data:font/woff2;base64,{encoded}) format('woff2');}}"
        )
    return '<style>\n' + '\n'.join(rules) + '\n</style>'


def finalize(markup: str, view_box: str, style: str) -> str:
    """Add xmlns/width/height and insert the font `<style>` into the SVG."""
    open_tag = re.match(r'<svg\b[^>]*>', markup)
    if open_tag is None:
        message = 'serialized markup does not start with an <svg> tag'
        raise RuntimeError(message)
    tag = open_tag.group(0)
    extra = ''
    if 'xmlns=' not in tag:
        extra += ' xmlns="http://www.w3.org/2000/svg"'
    width, height = view_box.split()[2:4]
    tag = re.sub(r'\s+(width|height)="[^"]*"', '', tag)
    tag = tag[:-1] + f'{extra} width="{width}" height="{height}">'
    markup = tag + markup[open_tag.end() :]

    anchors = list(re.finditer(r'</(?:title|desc)>', markup))
    at = anchors[-1].end() if anchors else len(tag)
    markup = markup[:at] + '\n' + style + markup[at:]
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + markup + '\n'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stix = stix_sources()
    launch = {'executable_path': sys.argv[1]} if len(sys.argv) > 1 else {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page()
        for src in sorted(HERE.glob('*.html')):
            for theme in THEMES:
                page.goto(f'file://{src}')
                page.evaluate(
                    '(theme) => { if (theme === "dark") '
                    '{ document.documentElement.dataset.theme = "dark"; } '
                    'else { delete document.documentElement.dataset.theme; } }',
                    theme,
                )
                page.wait_for_load_state('networkidle')
                page.evaluate('() => document.fonts.ready')
                collected = page.evaluate(COLLECT_JS)
                if 'var(' in collected['markup']:
                    message = f'{src.name} ({theme}): unresolved var() left in the SVG'
                    raise RuntimeError(message)
                svg = finalize(
                    collected['markup'],
                    collected['viewBox'],
                    font_style_block(collected['usage'], stix),
                )
                out = OUT / f'{src.stem}-{theme}.svg'
                out.write_text(svg, encoding='utf-8')
                print(f'{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.1f} KB')
        browser.close()


if __name__ == '__main__':
    main()
