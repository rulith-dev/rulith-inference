#pragma once

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mutex>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

struct llama_lazy_reader {
#ifdef _WIN32
    using file_handle = HANDLE;
    struct io_context {
        HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        io_context() {
            if (!event) {
                throw std::runtime_error(format("lazy read event failed: %lu", (unsigned long) GetLastError()));
            }
        }
        ~io_context() { CloseHandle(event); }
        io_context(const io_context &) = delete;
        io_context & operator=(const io_context &) = delete;
    };

    void read_at(size_t offset, uint8_t * dst, size_t count, io_context & io) const {
        for (size_t done = 0; done < count; ) {
            const uint64_t pos = (uint64_t) offset + done;
            OVERLAPPED request = {};
            request.Offset = (DWORD) pos;
            request.OffsetHigh = (DWORD) (pos >> 32);
            request.hEvent = io.event;
            const DWORD size = (DWORD) std::min<size_t>(count - done, MAXDWORD);
            if (!ReadFile(fd, dst + done, size, nullptr, &request) && GetLastError() != ERROR_IO_PENDING) {
                throw std::runtime_error(format("lazy read at %zu failed: %lu", offset + done, (unsigned long) GetLastError()));
            }
            DWORD received = 0;
            if (!GetOverlappedResult(fd, &request, &received, TRUE)) {
                throw std::runtime_error(format("lazy read completion at %zu failed: %lu", offset + done, (unsigned long) GetLastError()));
            }
            if (received == 0) {
                throw std::runtime_error(format("lazy read at %zu reached unexpected EOF", offset + done));
            }
            done += received;
        }
    }

    template <typename F>
    void launch_prefetch(F && work) const {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        if (prefetch_thread.joinable()) {
            prefetch_thread.join();
        }
        prefetch_thread = std::thread([fn = std::forward<F>(work)]() {
            try {
                fn();
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("lazy prefetch skipped: %s\n", e.what());
            } catch (...) {
                LLAMA_LOG_WARN("lazy prefetch skipped after an exception\n");
            }
        });
    }

    mutable std::mutex prefetch_mutex;
    mutable std::thread prefetch_thread;

    // strixllama: keep several overlapped reads in flight per worker instead of one synchronous
    // read_at per row. Depth 1 measured badly on both gather shapes (scripts/hip-bench.ps1 -Trace):
    // the 16-row decode gather ran single-threaded at 3-7 ms per token, and the 262144-row prefill
    // gather spent ~750 us per row per worker against a 172 us disk latency (tools/ple_io_depth.py
    // saw a queue depth of ~6 from 64 workers). Row order within a worker's range is preserved by
    // finishing slots round-robin in issue order.
    static constexpr int STRIX_IO_DEPTH = 16;

    struct inflight {
        io_context io;
        OVERLAPPED ov{};
        std::vector<uint8_t> buf;
        int64_t i = -1, j = -1;   // pairs[i..j] all name this row
        bool pending = false;
    };

    void run_range_overlapped(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                              int64_t begin, int64_t end, float * dst) const {
        std::vector<inflight> slots((size_t) std::min<int64_t>(STRIX_IO_DEPTH, std::max<int64_t>(1, end - begin)));
        for (auto & s : slots) { s.buf.resize(row_size); }
        int64_t next = begin;
        int live = 0;

        auto issue = [&](inflight & s) {
            const int64_t i = next;
            int64_t j = i;
            while (j + 1 < end && pairs[j + 1].first == pairs[i].first) { ++j; }
            s.i = i; s.j = j; next = j + 1;
            const uint64_t pos = (uint64_t) base + (uint64_t) pairs[i].first * row_size;
            s.ov = {};
            s.ov.Offset     = (DWORD) pos;
            s.ov.OffsetHigh = (DWORD) (pos >> 32);
            s.ov.hEvent     = s.io.event;
            ResetEvent(s.io.event);
            if (!ReadFile(fd, s.buf.data(), (DWORD) row_size, nullptr, &s.ov) && GetLastError() != ERROR_IO_PENDING) {
                throw std::runtime_error(format("lazy read at %llu failed: %lu", (unsigned long long) pos, (unsigned long) GetLastError()));
            }
            s.pending = true; ++live;
        };

        auto finish = [&](inflight & s) {
            DWORD got = 0;
            if (!GetOverlappedResult(fd, &s.ov, &got, TRUE)) {
                throw std::runtime_error(format("lazy read completion failed: %lu", (unsigned long) GetLastError()));
            }
            if (got < row_size) {   // a short read is legal; finish the remainder synchronously
                const size_t off = base + (size_t) pairs[s.i].first * row_size;
                read_at(off + got, s.buf.data() + got, row_size - got, s.io);
            }
            float * first = dst + (size_t) pairs[s.i].second * head_dim;
            if (to_float) {
                to_float(s.buf.data(), first, head_dim);
            } else {
                memcpy(first, s.buf.data(), (size_t) head_dim * sizeof(float));
            }
            for (int64_t k = s.i + 1; k <= s.j; ++k) {
                memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
            }
            s.pending = false; --live;
        };

        for (auto & s : slots) { if (next >= end) { break; } issue(s); }
        size_t rr = 0;
        while (live > 0) {
            inflight & s = slots[rr];
            if (s.pending) {
                finish(s);
                if (next < end) { issue(s); }
            }
            rr = (rr + 1) % slots.size();
        }
    }

#else
    using file_handle = int;
#endif
    llama_lazy_reader(file_handle fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
#ifdef _WIN32
        if (prefetch_thread.joinable()) {
            prefetch_thread.join();
        }
        if (fd != INVALID_HANDLE_VALUE) {
            CloseHandle(fd);
        }
#else
        if (fd >= 0) {
            ::close(fd);
        }
#endif
    }

    const file_handle fd;
    const size_t   base;       // file offset of row 0
    const size_t   row_size;   // bytes per quantized row
    const int64_t  n_rows;
    const int      n_threads;  // in-flight read workers
    const int64_t  head_dim;
    ggml_to_float_t to_float;  // same dequantizer the ggml_get_rows CPU kernel uses

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst) const {
        // STRIX_PLE_TRACE=1 times every gather, so the gather's share of a prefill is a
        // measurement rather than an inference from disk counters (which average over its bursts).
        const bool strixllama_trace = [] {
            const char * t = getenv("STRIX_PLE_TRACE");
            return t && atoi(t);
        }();
        const auto strixllama_t0 = std::chrono::steady_clock::now();
        std::vector<std::pair<int32_t, int32_t>> pairs;
        pairs.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        // small gathers are not worth a thread per row
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, n / 32));

        auto run_chunk = [&](int w, std::exception_ptr & err) {
            try {
                run_range(pairs, n * w / n_workers, n * (w + 1) / n_workers, dst);
            } catch (...) {
                err = std::current_exception();
            }
        };

        std::vector<std::exception_ptr> errs(n_workers);
        std::vector<std::thread> workers;
        try {
            for (int w = 1; w < n_workers; ++w) {
                workers.emplace_back([&run_chunk, &errs, w]() {
                    run_chunk(w, errs[w]);
                });
            }
        } catch (...) {
            for (auto & t : workers) {
                t.join();
            }
            throw;
        }

        run_chunk(0, errs[0]);
        for (auto & t : workers) {
            t.join();
        }

        for (const auto & err : errs) {
            if (err) {
                std::rethrow_exception(err);
            }
        }

        if (strixllama_trace) {
            const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - strixllama_t0).count();
            int64_t uniq = n ? 1 : 0;
            for (int64_t i = 1; i < n; ++i) {
                if (pairs[i].first != pairs[i - 1].first) { ++uniq; }
            }
            fprintf(stderr, "PLE_GATHER rows=%lld uniq=%lld workers=%d row_bytes=%zu ms=%.1f\n",
                    (long long) n, (long long) uniq, n_workers, row_size, ms);
        }
    }

    // populate the page cache for the rows a later gather() will read; never writes any output, so a wrong
    // prediction only wastes readahead. Safe to call concurrently with gather().
    void prefetch(const int32_t * rows, int64_t n) const {
        std::vector<int32_t> uniq;
        uniq.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            if (rows[i] >= 0 && (int64_t) rows[i] < n_rows) { uniq.push_back(rows[i]); }
        }
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, (int64_t) uniq.size() / 32));
        auto run = [&](int w) {
#ifdef _WIN32
            try {
            io_context io;
            std::vector<uint8_t> bounce(row_size);
#endif
            const int64_t b = (int64_t) uniq.size() * w / n_workers, e = (int64_t) uniq.size() * (w + 1) / n_workers;
            for (int64_t i = b; i < e; ++i) {
                const size_t off = base + (size_t) uniq[i] * row_size;
#ifdef _WIN32
                read_at(off, bounce.data(), row_size, io);
#else
                ::posix_fadvise(fd, (off_t) off, (off_t) row_size, POSIX_FADV_WILLNEED);
#endif
            }
#ifdef _WIN32
            } catch (...) {
                // A prefetch failure must not fail a later gather.
            }
#endif
        };
        std::vector<std::thread> workers;
        try {
            for (int w = 1; w < n_workers; ++w) { workers.emplace_back([&run, w]() { run(w); }); }
        } catch (...) {
            for (auto & t : workers) { t.join(); }
            return;
        }
        run(0);
        for (auto & t : workers) { t.join(); }
    }

private:
    void run_range(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                   int64_t begin, int64_t end, float * dst) const {
#ifdef _WIN32
        run_range_overlapped(pairs, begin, end, dst);
#else
        std::vector<uint8_t> bounce(row_size);
        for (int64_t i = begin; i < end; ) {
            int64_t j = i;
            while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
                ++j;
            }
            const size_t off = base + (size_t) pairs[i].first * row_size;
            for (size_t done = 0; done < row_size; ) {
                const ssize_t n_read = ::pread(fd, bounce.data() + done, row_size - done, off + done);
                if (n_read < 0 && errno == EINTR) {
                    continue;
                }
                if (n_read <= 0) {
                    throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                            row_size, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
                }
                done += n_read;
            }
            float * first = dst + (size_t) pairs[i].second * head_dim;
            if (to_float) {
                to_float(bounce.data(), first, head_dim);
            } else {
                memcpy(first, bounce.data(), (size_t) head_dim * sizeof(float));
            }
            for (int64_t k = i + 1; k <= j; ++k) {
                memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
            }
            i = j + 1;
        }
#endif
    }
};
