#!/usr/bin/env python3
"""Give the upstream strix-halo tree a working Windows PLE direct reader.

Upstream only implements `--lazy-mode on-direct` for POSIX, in two places:

  1. llama_lazy_reader's `#ifdef _WIN32` branch is a stub whose gather() calls GGML_ABORT, and
     prefetch() exists only in the POSIX branch, so since 40a9f4d0 the tree does not even compile
     on Windows (qwen4exp.cpp: no member named 'prefetch' in 'llama_lazy_reader').
  2. llama_model_base::load_lazy_reader has a whole second definition under `#else` for Windows
     that logs "--lazy-mode on-direct is not supported on this platform, using lazy mmap reads"
     and returns nullptr. That warning carries neither "error" nor "failed", so a filtered
     benchmark log never shows it - and the mmap fallback it causes has no prefetch and no
     concurrency, which measured 216 t/s against 850 for the direct reader.

This carries our cross-platform reader forward and makes the real load_lazy_reader compile on
every platform, opening the file with CreateFileW on Windows. The POSIX paths are byte-identical
to upstream's, so nothing upstream does to them is lost. Idempotent; each step checks itself.

Usage: python patches/apply_win_lazy_reader.py [tree]
"""
import io
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The reader is vendored here rather than copied out of a sibling checkout: this repository ships
# no llama.cpp tree, and the patch has to work against whatever bootstrap.py just cloned.
DONOR = os.path.join(ROOT, "patches", "lazy-reader", "llama-lazy-reader.h")
tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")

READER = os.path.join(tree, "src", "llama-lazy-reader.h")
MODEL = os.path.join(tree, "src", "llama-model.cpp")

OLD_OPEN = """    // an independently opened buffered descriptor: dup() would share the
    // loader's open file description, whose readahead advice and O_DIRECT
    // flag would fight the small scattered row reads
    const int fd = ::open(ml.files[w->idx]->name().c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LLAMA_LOG_WARN("%s: could not open %s for direct reads (%s), using lazy mmap reads\\n",
                __func__, ml.files[w->idx]->name().c_str(), strerror(errno));
        return nullptr;
    }

#ifdef __linux__
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
"""

NEW_OPEN = """#ifdef _WIN32
    // CreateFileW, not ::open: the reader issues overlapped reads against this handle
    const std::string & name = ml.files[w->idx]->name();
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.c_str(), -1, nullptr, 0);
    if (size == 0) {
        throw std::runtime_error("invalid UTF-8 path for lazy reader");
    }
    std::wstring wide(size, L'\\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.c_str(), -1, wide.data(), size)) {
        throw std::runtime_error("failed to convert lazy reader path");
    }
    const HANDLE fd = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (fd == INVALID_HANDLE_VALUE) {
        LLAMA_LOG_WARN("%s: Windows lazy reader open failed (%lu), using lazy mmap reads\\n",
                __func__, (unsigned long) GetLastError());
        return nullptr;
    }
#else
    // an independently opened buffered descriptor: dup() would share the
    // loader's open file description, whose readahead advice and O_DIRECT
    // flag would fight the small scattered row reads
    const int fd = ::open(ml.files[w->idx]->name().c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        LLAMA_LOG_WARN("%s: could not open %s for direct reads (%s), using lazy mmap reads\\n",
                __func__, ml.files[w->idx]->name().c_str(), strerror(errno));
        return nullptr;
    }

#ifdef __linux__
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif

#endif
"""

# the real definition was POSIX-only; the guard goes and so does the Windows stub after it
OLD_GUARD = """#ifndef _WIN32
const llama_lazy_reader * llama_model_base::load_lazy_reader(llama_model_loader & ml, const char * tensor_name, const ggml_tensor * t) {
"""
NEW_GUARD = """// strixllama: this definition is compiled on every platform; the Windows open lives in its
// _WIN32 branch below instead of in a separate stub that always fell back to mmap reads
const llama_lazy_reader * llama_model_base::load_lazy_reader(llama_model_loader & ml, const char * tensor_name, const ggml_tensor * t) {
"""

OLD_STUB = """    lazy_readers[tensor_name] = std::move(reader);
    return lazy_readers.at(tensor_name).get();
}
#else
const llama_lazy_reader * llama_model_base::load_lazy_reader(llama_model_loader & ml, const char *, const ggml_tensor *) {
    if (ml.lazy.mode == LLAMA_LAZY_MODE_DIRECT) {
        LLAMA_LOG_WARN("%s: --lazy-mode on-direct is not supported on this platform, using lazy mmap reads\\n", __func__);
    }
    return nullptr;
}
#endif
"""
NEW_STUB = """    lazy_readers[tensor_name] = std::move(reader);
    return lazy_readers.at(tensor_name).get();
}
"""


def step(s, old, new, what):
    if new in s:
        print("already applied:", what)
        return s
    if s.count(old) != 1:
        sys.exit("anchor not found (%s) in %s" % (what, MODEL))
    print("patched:", what)
    return s.replace(old, new)


def main():
    for path in (DONOR, READER, MODEL):
        if not os.path.isfile(path):
            sys.exit("missing %s" % path)

    s = io.open(MODEL, encoding="utf-8").read()
    s = step(s, OLD_OPEN, NEW_OPEN, "CreateFileW open branch")
    s = step(s, OLD_GUARD, NEW_GUARD, "drop the #ifndef _WIN32 around the real definition")
    s = step(s, OLD_STUB, NEW_STUB, "drop the Windows stub definition")
    io.open(MODEL, "w", encoding="utf-8", newline="").write(s)

    # upstream's POSIX gather/run_range/prefetch are byte-identical to the donor's, so this only
    # adds the Windows branch back (plus the STRIX_PLE_TRACE gather timing if the donor has it)
    if io.open(DONOR, "rb").read() != io.open(READER, "rb").read():
        shutil.copyfile(DONOR, READER)
        print("installed the cross-platform reader into", READER)
    else:
        print("reader already in place")


if __name__ == "__main__":
    main()
