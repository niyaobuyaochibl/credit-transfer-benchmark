"""Reproduce the paper's primary transfer contrast from the released per-participant contrasts."""
import json
from pathlib import Path

import numpy as np

from transfer_check import transfer_contrast

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'paper_heldout_contrasts.json').read_text(encoding='utf-8'))
directions = ['summed->component-wise', 'component-wise->summed']
people = data['participants']
contrast = {d: np.array([p[d] for p in people]) for d in directions}
zero = {d: np.zeros(len(people)) for d in directions}
result = transfer_contrast(contrast, zero, [p['session_order'] for p in people], min_effect=0.005,
                           draws=data['bootstrap']['draws'], seed=data['bootstrap']['seed'])

# The file stores B* minus candidate; transfer_contrast reports baseline minus candidate, so the signs agree.
rows = [('primary', result['primary'])] + [(d, result['directions'][d]) for d in directions]
for name, r in rows:
    published = data['published'][name]
    match = np.isclose(r['mean'], published['mean'], rtol=0, atol=1e-12) and np.allclose(r['ci95'], published['ci95'], rtol=0, atol=1e-12)
    print(f"{name:24s} mean {r['mean']:+.4f}  95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  matches paper: {bool(match)}")
print('decision checks:', result['checks'], '-> candidate supported:', result['candidate_supported'])
