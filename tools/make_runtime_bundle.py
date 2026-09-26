"""Assemble a self-contained runtime: everything the desktop app needs besides the model files.

    python tools/make_runtime_bundle.py [--out dist/runtime] [--gfx gfx1151] [--proxy http://127.0.0.1:7897]

The result is what the installer ships under <install dir>/runtime:

    bin/hip/           llama-server and its DLLs, the ROCm DLLs it imports (found by walking the PE
                       import tables, not by a list that goes stale), the rocBLAS / hipBLASLt kernel
                       libraries for one GPU, and the Visual C++ and OpenMP runtimes
    tools/manager.py   the manager, unchanged
    python/            CPython's embeddable distribution - the manager is standard library only
    BUNDLE.json        what went in, from where, and the upstream pin it was built from

Nothing here is specific to the machine it was made on except the GPU: the kernel libraries are
copied for --gfx only, which is what keeps the bundle at a few hundred MB instead of gigabytes.
"""
import argparse
import datetime as dt
import glob
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# bootstrap.py --build installs here (bin/hip-rocm101 up to 0.2.4, which a tree built before still has)
RUNTIME_DIR = next((d for d in (ROOT / 'bin' / 'hip', ROOT / 'bin' / 'hip-rocm101') if (d / 'llama-server.exe').is_file()),
                   ROOT / 'bin' / 'hip')
ROCM_BIN = ROOT / 'toolchain' / 'rocm-venv' / 'Lib' / 'site-packages' / '_rocm_sdk_devel' / 'bin'
PYTHON_EMBED = ('3.12.10', 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip')
# what llama-server needs from its own build directory: the server, its implementation, and the
# libraries they load. The bench, perplexity and quantize tools are not shipped.
OWN = ('llama-server.exe', 'llama-server-impl.dll', 'llama-common.dll', 'llama.dll', 'mtmd.dll',
       'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-hip.dll')
VC_RUNTIME = ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')
OPENMP = 'libomp140.x86_64.dll'      # ggml-base and ggml-cpu import it: this build's clang-cl links LLVM OpenMP


def pe_imports(path):
    """The DLL names in a PE file's import table."""
    d = path.read_bytes()
    pe = struct.unpack_from('<I', d, 0x3c)[0]
    nsec = struct.unpack_from('<H', d, pe + 6)[0]
    optsz = struct.unpack_from('<H', d, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from('<H', d, opt)[0]
    directories = opt + (112 if magic == 0x20b else 96)
    import_rva = struct.unpack_from('<I', d, directories + 8)[0]
    sections = [struct.unpack_from('<8sIIII', d, opt + optsz + 40 * i) for i in range(nsec)]

    def offset(rva):
        for _, vsize, vaddr, rsize, raw in sections:
            if vaddr <= rva < vaddr + max(vsize, rsize):
                return rva - vaddr + raw
        raise ValueError(f'{path.name}: import table outside every section')

    names = []
    if not import_rva:
        return names
    o = offset(import_rva)
    while True:
        _, _, _, name_rva, _ = struct.unpack_from('<IIIII', d, o)
        o += 20
        if not name_rva:
            return names
        n = offset(name_rva)
        names.append(d[n:d.index(b'\0', n)].decode())


def closure(roots, search):
    """Every module in `search` that `roots` load, directly or through other modules in `search`."""
    found, todo, seen = {}, [r.lower() for r in roots], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        path = search.get(name)
        if path is None:
            continue
        found[name] = path
        todo.extend(dep.lower() for dep in pe_imports(path))
    return found


def vs_redist(name, subdir):
    """A DLL from the newest Visual Studio redistributable directory that has it."""
    pattern = r'C:\Program Files*\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\%s\%s' % (subdir, name)
    hits = sorted(glob.glob(pattern))
    return Path(hits[-1]) if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=str(ROOT / 'dist' / 'runtime'))
    ap.add_argument('--gfx', default='gfx1151', help='GPU whose kernel libraries are shipped')
    ap.add_argument('--proxy', default=os.environ.get('HTTPS_PROXY', ''), help='HTTP proxy for the Python download')
    ap.add_argument('--python-zip', default='', help='an already downloaded embeddable zip, instead of fetching one')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if ROOT / 'dist' not in out.parents:
        sys.exit(f'--out must be under {ROOT / "dist"}: it is wiped first')
    if not (RUNTIME_DIR / 'llama-server.exe').is_file():
        sys.exit(f'no runtime at {RUNTIME_DIR}; run bootstrap/bootstrap.py first')
    if not (ROCM_BIN / 'amdhip64_7.dll').is_file():
        sys.exit(f'no ROCm SDK at {ROCM_BIN}')
    shutil.rmtree(out, ignore_errors=True)
    bin_out = out / 'bin' / 'hip'
    bin_out.mkdir(parents=True)
    record = dict(built_at=dt.datetime.now().astimezone().isoformat(), gfx=args.gfx, files={})

    def take(src, dest, origin):
        shutil.copy2(src, dest)
        record['files'][str(dest.relative_to(out)).replace('\\', '/')] = {'bytes': src.stat().st_size, 'from': origin}

    for name in OWN:
        take(RUNTIME_DIR / name, bin_out / name, 'bin/' + RUNTIME_DIR.name)
    rocm = {p.name.lower(): p for p in ROCM_BIN.iterdir() if p.suffix.lower() == '.dll'}
    own = {n.lower(): RUNTIME_DIR / n for n in OWN}
    for name, path in sorted(closure(list(OWN), {**rocm, **own}).items()):
        if name in rocm:
            take(path, bin_out / path.name, 'rocm sdk bin')
    # kernel libraries: rocBLAS and hipBLASLt look for them next to their DLL, in <name>/library
    for lib in ('rocblas', 'hipblaslt'):
        src = ROCM_BIN / lib / 'library'
        if not src.is_dir():
            continue
        (bin_out / lib / 'library').mkdir(parents=True)
        # one subdirectory per architecture, plus files shared by all of them
        for entry in sorted(src.iterdir()):
            if 'gfx' in entry.name and args.gfx not in entry.name:
                continue
            for f in ([entry] if entry.is_file() else sorted(p for p in entry.rglob('*') if p.is_file())):
                dest = bin_out / lib / 'library' / f.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                take(f, dest, f'rocm sdk bin/{lib}/library')
    for name in VC_RUNTIME:
        src = vs_redist(name, 'Microsoft.VC143.CRT')
        if not src:
            sys.exit(f'{name}: no Visual C++ redistributable found under Visual Studio')
        take(src, bin_out / name, 'Visual C++ redistributable')
    src = vs_redist(OPENMP, 'Microsoft.VC143.OpenMP.LLVM')
    if not src:
        # the release runtime Visual Studio installs into System32 is that same package; the copy
        # under debug_nonredist is not, and is never used here
        src = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / OPENMP
        if not src.is_file():
            sys.exit(f'{OPENMP}: install the "C++ OpenMP LLVM runtime" Visual Studio component')
        print(f'note: {OPENMP} taken from System32 (no redistributable copy under Visual Studio)')
    take(src, bin_out / OPENMP, 'Visual C++ OpenMP (LLVM) runtime')

    (out / 'tools').mkdir()
    take(ROOT / 'tools' / 'manager.py', out / 'tools' / 'manager.py', 'tools')
    for d in ('config', 'logs', 'models'):
        (out / d).mkdir()

    version, url = PYTHON_EMBED
    if args.python_zip:
        data = Path(args.python_zip).read_bytes()
    else:
        handler = urllib.request.ProxyHandler({'https': args.proxy, 'http': args.proxy} if args.proxy else {})
        print('downloading', url)
        with urllib.request.build_opener(handler).open(url, timeout=600) as r:
            data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(out / 'python')
    record['python'] = dict(version=version, source=url, bytes=len(data))
    probe = subprocess.run([str(out / 'python' / 'python.exe'), '-c',
                            'import ctypes, winreg, json, msvcrt, urllib.request, hashlib, uuid; print("ok")'],
                           capture_output=True, text=True)
    if probe.stdout.strip() != 'ok':
        sys.exit(f'embedded Python cannot import what the manager needs: {probe.stderr}')

    pin = json.loads((ROOT / 'bootstrap' / 'UPSTREAM.json').read_text(encoding='utf-8'))
    record['upstream'] = {k: pin[k] for k in ('repo', 'revision') if k in pin}
    rocm_info = sorted(glob.glob(str(ROCM_BIN.parents[1] / 'rocm_sdk_devel*dist-info')))
    record['rocm'] = Path(rocm_info[-1]).name if rocm_info else 'unknown'
    total = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    record['total_bytes'] = total
    (out / 'BUNDLE.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'{out}: {len(record["files"])} files listed, {total / 2**20:.0f} MiB in all; rocm={record["rocm"]}')


if __name__ == '__main__':
    main()
