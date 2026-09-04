"""Write split-overview.html from the data produced by split-overview.jl.

Run `julia --project=. docs/diagrams/split-overview.jl` first; it leaves
`docs/diagrams/.cache/split-overview.json` behind. This script lays the three
scatter panels out with the shared theme tokens, so `export.py` can render
the light and dark SVGs like the other figures in this directory.

    python docs/diagrams/split-overview.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / '.cache' / 'split-overview.json'
OUT = HERE / 'split-overview.html'

FONT = "Pretendard, Helvetica, 'Helvetica Neue', 'Nimbus Sans', Arial, sans-serif"
MATH = "'STIX Two Text', 'STIX Two Math', 'Times New Roman', serif"
PANEL_WIDTH, PANEL_HEIGHT, PADDING = 280, 220, 18
TOP = 64
PANEL_X = (40, 340, 640)


def main() -> None:
    """Render the figure source."""
    data = json.loads(DATA.read_text())
    xs, ys = data['x'], data['y']
    n_rows = len(xs)
    split_test = set(data['split_test'])
    rand_test = set(data['rand_test'])
    n_test = len(split_test)

    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    scale = min(
        (PANEL_WIDTH - 2 * PADDING) / (x_max - x_min),
        (PANEL_HEIGHT - 2 * PADDING) / (y_max - y_min),
    )

    def point(i: int, x0: int) -> tuple[str, str]:
        cx = x0 + PANEL_WIDTH / 2 + (xs[i] - (x_min + x_max) / 2) * scale
        cy = TOP + PANEL_HEIGHT / 2 - (ys[i] - (y_min + y_max) / 2) * scale
        return f'{cx:.1f}', f'{cy:.1f}'

    def panel(x0: int, label: str, test: set[int]) -> str:
        lines = [
            f'  <rect x="{x0}" y="{TOP}" width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" rx="4" '
            'fill="var(--surface)" stroke="var(--ink-a12)" stroke-width="1"/>',
            f'  <text x="{x0}" y="{TOP - 14}" fill="var(--muted)" font-size="9" '
            f'font-family="{FONT}" letter-spacing="0.12em">{label}</text>',
        ]
        for i in range(n_rows):
            if i + 1 in test:
                continue
            cx, cy = point(i, x0)
            lines.append(
                f'  <circle cx="{cx}" cy="{cy}" r="3" fill="var(--muted-a20)" '
                'stroke="var(--muted)" stroke-width="0.9"/>'
            )
        for i in range(n_rows):
            if i + 1 in test:
                cx, cy = point(i, x0)
                lines.append(f'  <circle cx="{cx}" cy="{cy}" r="3.6" fill="var(--accent)"/>')
        return '\n'.join(lines)

    def corner(x0: int, text: str) -> str:
        return (
            f'  <text x="{x0 + PANEL_WIDTH}" y="{TOP - 14}" fill="var(--soft)" font-size="10" '
            f'font-family="{MATH}" text-anchor="end">{text}</text>'
        )

    def caption(x0: int, value: float) -> str:
        return (
            f'  <text x="{x0 + PANEL_WIDTH / 2}" y="{TOP + PANEL_HEIGHT + 22}" '
            f'fill="var(--muted)" font-size="10" font-family="{FONT}" text-anchor="middle">'
            'energy distance between the sides '
            f'<tspan fill="var(--ink)" font-weight="600">{value:.3f}</tspan></text>'
        )

    def arrow(x0: int, x1: int) -> str:
        y = TOP + PANEL_HEIGHT / 2
        return (
            f'  <line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="var(--muted)" '
            'stroke-width="1.2" marker-end="url(#arrow)"/>'
        )

    test_rows = f'<tspan font-style="italic">n</tspan> = {n_test} test rows'
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Distribution-preserving split</title>
<link href="https://fonts.googleapis.com/css2?family=STIX+Two+Text:ital,wght@0,400;1,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="theme.css">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--paper); }}
  svg {{ width: 100%; min-width: 720px; display: block; }}
</style>
</head>
<body>
<svg viewBox="0 0 960 380" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="split-overview-title split-overview-desc">
  <title id="split-overview-title">Distribution-preserving split</title>
  <desc id="split-overview-desc">Three scatter plots of the same {n_rows} two-dimensional rows. Left: all rows. Middle: a random draw of {n_test} test rows, which lands unevenly across the three clusters. Right: the {n_test} test rows chosen by SPlit, spread through every cluster in proportion, with a much smaller energy distance between the two sides.</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="var(--muted)"/></marker>
  </defs>

  <rect width="100%" height="100%" fill="var(--paper)"/>

  <!-- ===================== panels ===================== -->
{panel(PANEL_X[0], 'ALL ROWS', set())}
{corner(PANEL_X[0], f'<tspan font-style="italic">N</tspan> = {n_rows}')}
{panel(PANEL_X[1], 'RANDOM SPLIT', rand_test)}
{corner(PANEL_X[1], test_rows)}
{panel(PANEL_X[2], 'SPLIT', split_test)}
{corner(PANEL_X[2], test_rows)}

  <!-- ===================== captions ===================== -->
{caption(PANEL_X[1], data['q_rand'])}
{caption(PANEL_X[2], data['q_split'])}

  <!-- ===================== arrows between panels ===================== -->
{arrow(PANEL_X[0] + PANEL_WIDTH + 6, PANEL_X[1] - 8)}
{arrow(PANEL_X[1] + PANEL_WIDTH + 6, PANEL_X[2] - 8)}

  <!-- ===================== editorial aside ===================== -->
  <text x="920" y="30" fill="var(--muted)" font-size="13" font-style="italic" font-family="{FONT}" text-anchor="end">the held-out rows follow the shape of the data, cluster by cluster</text>

  <!-- ===================== legend strip ===================== -->
  <line x1="40" y1="338" x2="920" y2="338" stroke="var(--ink-a12)" stroke-width="0.8"/>
  <text x="40" y="366" fill="var(--muted)" font-size="8" font-family="{FONT}" letter-spacing="0.12em">LEGEND</text>

  <circle cx="160" cy="362" r="3" fill="var(--muted-a20)" stroke="var(--muted)" stroke-width="0.9"/>
  <text x="172" y="366" fill="var(--muted)" font-size="9" font-family="{FONT}" letter-spacing="0.10em">TRAIN ROW</text>

  <circle cx="300" cy="362" r="3.6" fill="var(--accent)"/>
  <text x="312" y="366" fill="var(--muted)" font-size="9" font-family="{FONT}" letter-spacing="0.10em">TEST ROW</text>

  <text x="920" y="366" fill="var(--soft)" font-size="9" font-family="{FONT}" text-anchor="end">random split: the seed whose energy distance is the median of 200 draws</text>
</svg>
</body>
</html>
'''
    OUT.write_text(html)
    print(OUT.relative_to(HERE.parent.parent))


if __name__ == '__main__':
    main()
