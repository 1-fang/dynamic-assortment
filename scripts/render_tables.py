"""Render the released manuscript table cells as Markdown."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display(cell):
    if isinstance(cell, dict):
        return f"{Decimal(cell['mean']):,} ± {Decimal(cell['sd']):,}"
    return cell.replace('|', '\\|')


def render(table):
    lines = [f"# {table['title']}", '']
    for panel in table['panels']:
        columns = None
        start_block = True
        for row in panel:
            if 'heading' in row:
                lines.extend(['', f"**{row['heading']}**", ''])
                start_block = True
                continue
            cells = [display(cell) for cell in row['cells']]
            is_header = columns is None or (
                start_block and len(cells) == len(columns) and
                (cells[0] == columns[0] or cells[-1] == columns[-1]) and
                not any(isinstance(cell, dict) for cell in row['cells'])
            )
            if is_header:
                columns = cells
            if start_block:
                lines.append('| ' + ' | '.join(columns) + ' |')
                lines.append('| ' + ' | '.join(['---'] * len(columns)) + ' |')
                start_block = False
            if not is_header:
                lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    if table['notes']:
        lines.extend([table['notes'], ''])
    return '\n'.join(lines)


def rounded(value):
    return value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


def check_resource_rows(table):
    checked = 0
    for panel in table['panels']:
        for row in panel:
            cells = row.get('cells', [])
            means = [Decimal(cell['mean']) for cell in cells if isinstance(cell, dict)]
            if not means:
                continue
            base, full = means
            frozen = table['number'] == 'EC.16'
            difference = Decimal(cells[-2 if frozen else -1].replace(',', ''))
            if rounded(full - base) != difference:
                raise ValueError(f"Table {table['number']} inconsistent difference: {cells}")
            if frozen:
                uplift = Decimal(cells[-1].removesuffix('%'))
                if rounded(100 * (full - base) / base) != uplift:
                    raise ValueError(f"Table {table['number']} inconsistent uplift: {cells}")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', help='Number such as 3 or EC.16')
    args = parser.parse_args()
    entries = json.loads((ROOT / 'data/tables/index.json').read_text())
    selected = [item for item in entries if args.table in (None, item['number'])]
    if not selected:
        parser.error('Unknown table number')
    out = ROOT / 'out/tables'
    out.mkdir(parents=True, exist_ok=True)
    for item in selected:
        path = ROOT / item['data']
        table = json.loads(path.read_text())
        if table['number'] in {'7', '8', 'EC.16'}:
            expected = {'7': 16, '8': 25, 'EC.16': 41}[table['number']]
            if check_resource_rows(table) != expected:
                raise ValueError(f"Incomplete resource grid: {path.name}")
        (out / path.with_suffix('.md').name).write_text(render(table))
    print(f'Rendered {len(selected)} table exhibits in out/tables/')


if __name__ == '__main__':
    main()
