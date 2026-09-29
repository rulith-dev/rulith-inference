"""Create or update a GitHub release and upload its assets.

    python tools/publish_release.py v0.1.0 path/to/asset.exe [more assets...] [--notes notes.md] [--draft]

Uses the credential Git already holds for github.com (git credential fill), so nothing has to be
typed or stored anywhere new; the token never leaves this process. Assets that already exist on the
release with the same name are replaced. Honours HTTPS_PROXY.
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = 'rulith-dev/rulith-inference'


def credential():
    out = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                         capture_output=True, text=True, check=True).stdout
    fields = dict(line.split('=', 1) for line in out.strip().splitlines() if '=' in line)
    token = fields.get('password')
    if not token:
        sys.exit('no github.com credential in the Git credential store')
    return token


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tag')
    ap.add_argument('assets', nargs='*')
    ap.add_argument('--notes', help='markdown file with the release notes')
    ap.add_argument('--title', default='')
    ap.add_argument('--draft', action='store_true')
    ap.add_argument('--proxy', default=os.environ.get('HTTPS_PROXY', ''))
    args = ap.parse_args()
    token = credential()
    handler = urllib.request.ProxyHandler({'https': args.proxy, 'http': args.proxy} if args.proxy else {})
    opener = urllib.request.build_opener(handler)

    def call(url, method='GET', data=None, content_type='application/json'):
        req = urllib.request.Request(url, data=data, method=method, headers={
            'Accept': 'application/vnd.github+json', 'Authorization': 'Bearer ' + token,
            'User-Agent': 'strixllama-publish', 'Content-Type': content_type})
        try:
            with opener.open(req, timeout=3600) as r:
                body = r.read()
                return r.status, (json.loads(body) if body else {})
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b'{}')

    api = f'https://api.github.com/repos/{REPO}'
    status, release = call(f'{api}/releases/tags/{args.tag}')
    if status == 404:
        body = dict(tag_name=args.tag, name=args.title or args.tag, draft=args.draft,
                    body=Path(args.notes).read_text(encoding='utf-8') if args.notes else '')
        status, release = call(f'{api}/releases', 'POST', json.dumps(body).encode())
        if status != 201:
            sys.exit(f'creating the release failed: {status} {release.get("message")}')
        print('created', release['html_url'])
    elif status == 200:
        if args.notes:
            call(release['url'], 'PATCH', json.dumps(dict(body=Path(args.notes).read_text(encoding='utf-8'))).encode())
        print('exists', release['html_url'])
    else:
        sys.exit(f'looking up the release failed: {status} {release.get("message")}')

    existing = {a['name']: a for a in release.get('assets', [])}
    upload_url = release['upload_url'].split('{')[0]
    for asset in args.assets:
        path = Path(asset)
        if path.name in existing:
            call(existing[path.name]['url'], 'DELETE')
            print('replaced', path.name)
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        print(f'uploading {path.name} ({len(data) / 2**20:.0f} MiB)...', flush=True)
        status, uploaded = call(f'{upload_url}?name={urllib.parse.quote(path.name)}', 'POST', data, mime)
        if status != 201:
            sys.exit(f'upload of {path.name} failed: {status} {uploaded.get("message")}')
        print('  ', uploaded['browser_download_url'])


if __name__ == '__main__':
    main()
