"""Offline archive integrity and numerical consistency audit; not an inference run."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

root = Path(__file__).resolve().parents[1]
archive = root / 'evidence/14b/LittleLess_v081_14B_Evidence.zip'
assert archive.is_file(), f'Missing: {archive}'
print('ARCHIVE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    base = 'LittleLess_v081_14B/'
    manifest = json.loads(z.read(base + 'SHA256SUMS.json'))
    for name, expected in manifest.items():
        actual = hashlib.sha256(z.read(base + name)).hexdigest()
        assert actual == expected, f'SHA256 mismatch for {name}'
    r = json.loads(z.read(base + 'kv_4p5/result.json'))
    c = json.loads(z.read(base + 'kv_4p5/configuration.json'))
    monitor = json.loads(z.read(base + 'kv_4p5/monitor.json'))
    assert r['generated_tokens'] == 2 == len(r['rows'])
    assert len(c['checkpoint']) > 0 and c['shards_found'] == 8
    assert abs(r['sampled_peak_process_rss_mib'] - monitor['sampled_peak_process_rss_mib']) < 1e-6
    assert abs(r['tokens_per_second'] - 2/r['generation_seconds']) < 1e-7
    for i, row in enumerate(r['rows']):
        x = np.load(io.BytesIO(z.read(base + f'kv_4p5/logits/step{i}.npy')), allow_pickle=False)
        assert x.size == row['vocab_size'] == 152064
        assert np.isfinite(x).all()
        assert row['all_finite'] is True
print('PASS: internal 14B evidence SHA256, reported rate, RSS, logits finiteness and shape')
print('NOT VERIFIED: original model checkpoint, pinned model revision, parity, independent replay')
