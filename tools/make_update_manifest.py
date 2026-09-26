"""Write latest.json, the file the app's updater reads, for a signed release build.

    python tools/make_update_manifest.py dist/Strix-Llama_0.2.5_x64-setup.exe --notes dist/release-notes-v0.2.5.md

The app checks https://github.com/rulith-dev/strixllama/releases/latest/download/latest.json (the endpoint
integrations/jan/apply.py configures), so this file is uploaded with every release, next to the setup it
names. The setup's signature comes from the .sig file Tauri writes beside it when the build is signed
(TAURI_SIGNING_PRIVATE_KEY set); the app installs an update only if that signature verifies against the
public key it was built with (integrations/jan/updater.pub). The version is read from the setup's name.
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

REPO = 'rulith-dev/strixllama'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('setup', help='the signed setup, with its .sig beside it')
    ap.add_argument('--notes', help='release notes (markdown): the prompt shows the first paragraph')
    ap.add_argument('--out', help='where to write latest.json (default: beside the setup)')
    args = ap.parse_args()
    setup = Path(args.setup)
    sig = setup.with_name(setup.name + '.sig')
    if not setup.is_file() or not sig.is_file():
        sys.exit(f'need {setup} and {sig}: build with TAURI_SIGNING_PRIVATE_KEY set')
    m = re.search(r'_(\d+\.\d+\.\d+)_', setup.name)
    if not m:
        sys.exit(f'no version in {setup.name}')
    version = m.group(1)
    notes = ''
    if args.notes:
        text = Path(args.notes).read_text(encoding='utf-8').strip()
        notes = text.split('\n\n', 1)[0]
    url = f'https://github.com/{REPO}/releases/download/v{version}/{setup.name}'
    entry = {'signature': sig.read_text(encoding='utf-8').strip(), 'url': url}
    manifest = {
        'version': version,
        'notes': notes,
        'pub_date': dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        # the updater looks for <os>-<arch>-<installer> first and falls back to <os>-<arch>
        'platforms': {'windows-x86_64-nsis': entry, 'windows-x86_64': entry},
    }
    out = Path(args.out) if args.out else setup.with_name('latest.json')
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(f'{out}: {version} -> {url}')


if __name__ == '__main__':
    main()
