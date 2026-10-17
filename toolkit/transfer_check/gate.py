"""Step 3: hash-locked access to held-out data.

`freeze` records the SHA-256 of every file the held-out evaluation depends on (code,
fitted parameters, the development selection). `open_heldout` verifies those hashes
and appends to an access journal whose first entry binds the manifest; after the
first access the manifest can no longer be replaced. This documents, but cannot by
itself prove, that the held-out data were untouched before the freeze.
"""
import hashlib
import json
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Gate:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = self.folder / 'manifest.json'
        self.journal = self.folder / 'access_journal.jsonl'

    def freeze(self, paths, plan=None):
        if self.journal.exists():
            raise RuntimeError('held-out access has started; the manifest cannot be replaced')
        self.folder.mkdir(parents=True, exist_ok=True)
        files = {Path(p).resolve().as_posix(): sha256(p) for p in paths}
        manifest = {'frozen_unix': time.time(), 'files': files, 'plan': plan}
        self.manifest.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
        return manifest

    def verify(self):
        if not self.manifest.is_file():
            raise RuntimeError('no manifest: held-out evaluation is locked')
        manifest = json.loads(self.manifest.read_text(encoding='utf-8'))
        for path, expected in manifest['files'].items():
            if not Path(path).is_file():
                raise RuntimeError(f'frozen file missing: {path}')
            if sha256(path) != expected:
                raise RuntimeError(f'frozen file changed: {path}')
        if self.journal.exists():
            first = json.loads(self.journal.read_text(encoding='utf-8').splitlines()[0])
            if first['manifest_sha256'] != sha256(self.manifest):
                raise RuntimeError('manifest changed after first held-out access')
        return manifest

    def open_heldout(self, purpose):
        manifest = self.verify()
        entry = {'unix': time.time(), 'purpose': purpose, 'manifest_sha256': sha256(self.manifest)}
        with self.journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        return manifest

    def accesses(self):
        if not self.journal.exists():
            return []
        return [json.loads(line) for line in self.journal.read_text(encoding='utf-8').splitlines()]
