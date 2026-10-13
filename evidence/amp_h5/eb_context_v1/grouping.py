"""Train-only normalized global-edit components and deterministic 3-fold packing."""
from collections import Counter
import hashlib


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]


def grouped_folds(rows, *, guard=True):
    n = len(rows)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i in range(n):
        for j in range(i):
            a, b = rows[i]['sequence'], rows[j]['sequence']
            # Integer comparison fixes the threshold without float ambiguity.
            if 10 * (max(len(a),len(b)) - distance(a,b)) >= 7 * max(len(a),len(b)):
                edges.append([j, i])
                parent[root(i)] = root(j)
    components = {}
    for i in range(n):
        components.setdefault(root(i), []).append(i)
    groups = sorted(components.values(), key=lambda g: (-len(g), min(rows[i]['molecule_sha256'] for i in g)))
    counts = [Counter() for _ in range(3)]
    total = Counter(r['terminal_state'] for r in rows)
    assignment = [None] * n
    records = []
    for group in groups:
        gc = Counter(rows[i]['terminal_state'] for i in group)
        def cost(f):
            return (sum(((counts[k][t] + (gc[t] if k == f else 0)) / max(total[t],1))**2
                        for k in range(3) for t in ('FREE','AMD')), sum(counts[f].values()), f)
        fold = min(range(3), key=cost)
        counts[fold].update(gc)
        gid = hashlib.sha256('\n'.join(sorted(rows[i]['molecule_sha256'] for i in group)).encode()).hexdigest()
        for i in group:
            assignment[i] = fold
        records.append(dict(group_id=gid, indices=group, fold=fold, terminals=dict(gc)))
    assert all(assignment[a] == assignment[b] for a,b in edges)
    blockers = []
    if max(map(len, groups), default=0) > n / 2:
        blockers.append('largest component exceeds half of train')
    for t in ('FREE','AMD'):
        if sum(any(rows[i]['terminal_state'] == t for i in g) for g in groups) < 6:
            blockers.append(t + ': fewer than six represented components')
        if any(c[t] < 5 for c in counts):
            blockers.append(t + ': a held-out fold has fewer than five rows')
        if any(total[t] - c[t] < 1 for c in counts):
            blockers.append(t + ': a training fold lacks terminal support')
    report = dict(assignments=assignment, components=records, edges=edges,
                  folds=[dict(c) for c in counts], component_count=len(groups), blockers=blockers,
                  source_component_ids_used=False, threshold='normalized Levenshtein >= 0.70')
    if guard and blockers:
        raise GroupingBlocked(report)
    return report


class GroupingBlocked(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('; '.join(report['blockers']))


def synthetic_tests():
    assert distance('ACD', 'ACD') == 0
    assert distance('ACD', 'AD') == 1
    assert distance('AAA', 'CCC') == 3
    seqs = ['AAAAAAAAAA','AAAAAAAAAC','CCCCCCCCCC','CCCCCCCCCA', 'WWWWWWWWWW','GGGGGGGGGG']
    rows = [dict(sequence=s, terminal_state=('FREE' if i%2==0 else 'AMD'),
                 molecule_sha256=hashlib.sha256((s+str(i)).encode()).hexdigest()) for i,s in enumerate(seqs)]
    result = grouped_folds(rows, guard=False)
    assert result['assignments'][0] == result['assignments'][1]
    assert result['assignments'][2] == result['assignments'][3]
    assert sorted(i for g in result['components'] for i in g['indices']) == list(range(len(rows)))
    assert all(result['assignments'][a] == result['assignments'][b] for a,b in result['edges'])
    assert result == grouped_folds(rows, guard=False)
    # Exact sequences with different terminal labels remain together.
    rows[1]['sequence'] = rows[0]['sequence']
    result = grouped_folds(rows, guard=False)
    assert result['assignments'][0] == result['assignments'][1]
    return dict(status='PASS', tests=['edit boundaries','70pct components','cross-terminal duplicate',
                                    'every row once','no edge crosses folds','deterministic packing'])


if __name__ == '__main__':
    print(synthetic_tests())
