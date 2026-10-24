"""Verify a signed installer, update manifest, notes and optional recorded MTP comparisons.

Release-machine dependency: cryptography. No private signing key is read.

    python tools/verify_release_assets.py dist/Rulith-Inference_0.6.8_x64-setup.exe \
        --notes docs/releases/v0.6.8.md --sampled-dir tmp/rel068
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def verify(setup, notes, manifest_path, public_key_path, sampled_dir=None):
    match = re.search(r'_(\d+\.\d+\.\d+)_', setup.name)
    if not match:
        raise ValueError('installer filename has no version')
    version = match[1]
    payload = setup.read_bytes()
    signature_text = setup.with_name(setup.name + '.sig').read_text(encoding='utf-8').strip()
    public_text = public_key_path.read_text(encoding='utf-8').strip()
    public = base64.b64decode(base64.b64decode(public_text).decode().splitlines()[1], validate=True)
    lines = base64.b64decode(signature_text, validate=True).decode().splitlines()
    signature = base64.b64decode(lines[1], validate=True)
    if len(public) != 42 or len(signature) != 74 or public[2:10] != signature[2:10]:
        raise ValueError('signature key does not match updater public key')
    if signature[:2] not in (b'Ed', b'ED') or not lines[2].startswith('trusted comment: '):
        raise ValueError('unsupported signature format')
    key = Ed25519PublicKey.from_public_bytes(public[10:])
    message = hashlib.blake2b(payload).digest() if signature[:2] == b'ED' else payload
    key.verify(signature[10:], message)
    key.verify(base64.b64decode(lines[3], validate=True), signature[10:] + lines[2][17:].encode())

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    first_paragraph = notes.read_text(encoding='utf-8').strip().split('\n\n', 1)[0]
    if manifest['version'] != version or manifest['notes'] != first_paragraph:
        raise ValueError('manifest version or notes do not match the release')
    url = f'https://github.com/rulith-dev/rulith-inference/releases/download/v{version}/{setup.name}'
    for platform in ('windows-x86_64-nsis', 'windows-x86_64'):
        entry = manifest['platforms'][platform]
        if entry['signature'] != signature_text or entry['url'] not in (url, 'https://gh-proxy.com/' + url):
            raise ValueError(f'manifest signature or download URL differs: {platform}')

    pairs = 0
    if sampled_dir is not None:
        for model in ('305', '405'):
            for seed in ('1234', '1', '7'):
                name = f'smp_{model}_s{seed}'
                poll = (sampled_dir / f'{name}_p1.txt').read_bytes()
                sync = (sampled_dir / f'{name}_p0.txt').read_bytes()
                if not poll or poll != sync:
                    raise ValueError(f'sampled MTP outputs are empty or differ: {name}')
                pairs += 1
    return {'version': version, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'signature': 'verified', 'manifest': 'verified', 'sampled_pairs_equal': pairs}


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('setup', type=Path)
    ap.add_argument('--notes', required=True, type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--public-key', type=Path, default=root / 'integrations/jan/updater.pub')
    ap.add_argument('--sampled-dir', type=Path)
    args = ap.parse_args()
    try:
        result = verify(args.setup, args.notes, args.manifest or args.setup.with_name('latest.json'),
                        args.public_key, args.sampled_dir)
    except Exception as exc:
        sys.exit(f'release verification failed: {type(exc).__name__}: {exc}')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
