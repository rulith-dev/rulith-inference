"""Adapt Jan's tests to the product overlay without changing shipped application code."""
import hashlib
import json
import shutil
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def apply_test_overlay(tree):
    here = Path(__file__).resolve().parent
    tree = Path(tree)
    changes = []
    for entry in json.loads((here / 'tests/overlay.json').read_text(encoding='utf-8')):
        path = tree / entry['file']
        text = path.read_text(encoding='utf-8')
        if digest(text) == entry['after_sha256']:
            continue
        if digest(text) != entry['before_sha256']:
            raise RuntimeError(f'Jan test differs from the reviewed base: {path}')
        for old, new in entry['edits']:
            if text.count(old) != 1:
                raise RuntimeError(f'Jan test anchor is not unique: {path}')
            text = text.replace(old, new, 1)
        if digest(text) != entry['after_sha256']:
            raise RuntimeError(f'Jan test adaptation did not reproduce: {path}')
        changes.append((path, text))
    for path, text in changes:
        path.write_text(text, encoding='utf-8', newline='\n')
    # A worktree can share installed packages with another desktop. Keep test
    # transform caches in this checkout, beside its own generated route tree.
    config = tree / 'web-app/vitest.config.ts'
    source = config.read_text(encoding='utf-8')
    anchor = 'export default defineConfig({\n'
    cache = anchor + "  cacheDir: '.cache/vitest',\n"
    if cache not in source:
        if source.count(anchor) != 1:
            raise RuntimeError(f'Desktop test configuration changed: {config}')
        source = source.replace(anchor, cache)
    # The in-memory config loader runs this as ESM.
    if 'path.resolve(__dirname,' in source:
        source = source.replace("import path from 'path'", "import path from 'path'\nimport { fileURLToPath } from 'node:url'\nconst configDir = path.dirname(fileURLToPath(import.meta.url))")
        source = source.replace('path.resolve(__dirname,', 'path.resolve(configDir,')
    config.write_text(source, encoding='utf-8', newline='\n')
    for name in ('compact.test.ts', 'dates.test.ts', 'brainBridge.test.ts', 'BrainHost.test.tsx', 'brainNavigation.test.tsx'):
        shutil.copyfile(here / 'tests' / name, tree / 'web-app/src/components/strixllama' / name)
    print(f'Desktop tests: {len(changes)} files adapted; overlay tests installed')


if __name__ == '__main__':
    import sys
    apply_test_overlay(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / 'src/jan')
