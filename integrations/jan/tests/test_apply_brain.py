"""Exercise the generator against pristine, public desktop sources."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parents[1]
DESKTOP = HERE.parents[1] / 'src/jan'


def fingerprint(tree):
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob('*') if p.is_file()):
        digest.update(path.relative_to(tree).as_posix().encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.digest()


class DesktopOverlay(unittest.TestCase):
    def test_clean_apply_and_reapply(self):
        with tempfile.TemporaryDirectory(dir=DESKTOP.parent) as tmp:
            root = Path(tmp)
            archive = root / 'public.zip'
            subprocess.run(['git', '-c', 'safe.directory=' + DESKTOP.as_posix(), '-C', str(DESKTOP),
                            'archive', '--format=zip', '-o', str(archive), 'HEAD'], check=True)
            tree = root / 'desktop'
            with zipfile.ZipFile(archive) as source:
                source.extractall(tree)
            # Also works with the desktop's isolated embedded Python.
            apply = [sys.executable, '-B', '-c',
                     "import runpy,sys; p=sys.argv.pop(1); sys.path.insert(0,p); "
                     "runpy.run_path(p+'/apply.py',run_name='__main__')",
                     str(HERE), str(tree)]
            if not (HERE / 'brain_host.rs').exists():
                apply.append('--web-only')
            subprocess.run(apply, check=True, env={**os.environ, 'PYTHONUTF8': '1'})
            first = fingerprint(tree)
            subprocess.run(apply, check=True)
            self.assertEqual(first, fingerprint(tree))

            native = tree / 'src-tauri'
            web = tree / 'web-app/src'
            self.assertIn("createFileRoute('/brain')", (web / 'routes/brain.tsx').read_text())
            self.assertIn('<Link to="/brain">', (web / 'components/strixllama/Sidebar.tsx').read_text(encoding='utf-8'))
            security = json.loads((native / 'tauri.conf.json').read_bytes())['app']['security']
            self.assertEqual(security['csp']['frame-src'], 'https://brain-ui.rulith.invalid')
            self.assertIn('brain-host', security['capabilities'])
            cap = json.loads((native / 'capabilities/brain-host.json').read_bytes())
            self.assertEqual(cap['windows'], ['main'])
            self.assertTrue(cap['local'])
            self.assertNotIn('remote', cap)
            permission = json.loads((native / 'permissions/brain-host.json').read_bytes())['permission'][0]
            commands = permission['commands']['allow']
            self.assertEqual(set(commands), {'brain_host_' + name for name in (
                'mode', 'setup', 'mount', 'unmount', 'call', 'prepare_update')})
            legacy = json.loads((native / 'permissions/desktop-commands.json').read_bytes())['permission'][0]
            self.assertIn('settings_get', legacy['commands']['allow'])
            self.assertTrue(set(commands).isdisjoint(legacy['commands']['allow']))
            lib = (native / 'src/lib.rs').read_text(encoding='utf-8')
            for cmd in commands:
                self.assertEqual(lib.count(cmd + ','), 1, cmd)
            self.assertIn('request_saved_exit(app, 0)', lib)
            updater = (web / 'services/updater/tauri.ts').read_text(encoding='utf-8')
            self.assertLess(updater.index('await update.download(onProgress)'), updater.index("invoke('brain_host_prepare_update')"))
            self.assertLess(updater.index("invoke('brain_host_prepare_update')"), updater.index('await update.install()'))
            self.assertEqual(updater.count('await installUpdate(update'), 2)
            restart = (native / 'src/core/system/commands.rs').read_text(encoding='utf-8')
            self.assertEqual(restart.count('crate::prepare_desktop_exit('), 2)
            hooks = (native / 'windows/strixllama-hooks.nsh').read_text(encoding='utf-8')
            self.assertIn('!macro NSIS_HOOK_PREUNINSTALL', hooks)
            self.assertIn('--prepare-uninstall', hooks)
            self.assertNotIn('Stop-Process', hooks)
            self.assertNotIn('RMDir /r', hooks)


if __name__ == '__main__':
    unittest.main()
