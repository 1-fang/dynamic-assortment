"""Check the plot inputs against the displayed table cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ['Expedia30', 'Expedia Head', 'Expedia Regular', 'Expedia Cold',
          'KuaiSearch Overall', 'Q-Head', 'Q-Regular', 'Q-Cold', 'Q-Unseen']


def table_rows(number):
    path = ROOT / f'data/tables/table-{number}.json'
    return [r['cells'] for panel in json.loads(path.read_text())['panels']
            for r in panel if 'cells' in r]


def unpack(cell):
    return cell['mean'], cell['sd']


def reported(value):
    return tuple(x.strip().replace(',', '') for x in value.split('±'))


def matrix(number, method_column):
    by_method = {}
    for cells in table_rows(number):
        values = [unpack(cell) for cell in cells if isinstance(cell, dict)]
        if values:
            by_method.setdefault(cells[method_column], []).extend(values)
    return {(method, scene): value for method, values in by_method.items()
            for scene, value in zip(SCENES, values, strict=True)}


def resource_grid():
    data = {}
    for number, prefix, dimension in [('7', 'Expedia ', 'q'), ('8', 'KuaiSearch ', 'rho')]:
        for cells in table_rows(number):
            values = [unpack(cell) for cell in cells if isinstance(cell, dict)]
            if values:
                if cells[0]:
                    scene = cells[0] if cells[0] == 'Expedia30' or cells[0].startswith('Q-') else prefix+cells[0]
                data[scene, f'{dimension}={cells[1]}'] = tuple(values)
    return data


def main():
    normalized = json.loads((ROOT / 'data/current_results.json').read_text())
    count = 0
    for name, num in [('performance', '3'), ('ppo', '6')]:
        def normalize(method):
            return 'PPO + Qwen S/R' if method.startswith('PPO +') else method
        actual = {(normalize(m), s): value for (m, s), value in matrix(num, 0).items()}
        expected = {(r['method'], r['scene']): reported(r['reported']) for r in normalized[name]}
        if actual != expected:
            raise ValueError(f'Plot inputs differ from Table {num}: {set(actual.items()) ^ set(expected.items())}')
        count += len(expected)
    actual = matrix('4', 1)
    names = {'mBERT + mean pooling': 'mBERT', 'Sentence-Transformer': 'Sent.-Trans.', 'Multilingual E5': 'mE5', 'BGE': 'BGE-M3'}
    for name, *values in normalized['embedding_rows']['Transformer-Pointer']:
        count += len(values)
        assert all(actual[names[name], scene] == reported(value) for scene, value in zip(SCENES, values, strict=True))
    expected = {(r['scene'], r['level']): (reported(r['base']), reported(r['full'])) for r in normalized['resources']}
    assert resource_grid() == expected and len(expected) == 41
    count += 2 * len(expected)
    print(f'Verified {count} mean/SD cells against Tables 3, 4, 6, 7 and 8.')


if __name__ == '__main__':
    main()
