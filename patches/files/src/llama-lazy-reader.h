#pragma once

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <functional>
#include <mutex>
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

    // strixllama: runs after the prefetch before it, on its own thread, and never makes the caller wait for that one
    template <typename F>
    void launch_prefetch(F && work) const {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        std::thread prev = std::move(prefetch_thread);
        prefetch_thread = std::thread([prev = std::move(prev), fn = std::forward<F>(work)]() mutable {
            if (prev.joinable()) {
                prev.join();
            }
            try {
                fn();
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("lazy prefetch skipped: %s\n", e.what());
            } catch (...) {
                LLAMA_LOG_WARN("lazy prefetch skipped after an exception\n");
            }
        });
    }

    // strixllama: a pregather announced for the next batch waits until the batch about to run has its own rows
    // (launch_deferred, from the gather's input): started at once it shared the drive with that batch's gather, which
    // the GPU waits for. A newer announcement replaces one still waiting.
    void defer_prefetch(std::function<void()> work) const {
        std::lock_guard<std::mutex> lock(prefetch_mutex);
        deferred = std::move(work);
    }

    void launch_deferred() const {
        std::function<void()> work;
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex);
            work.swap(deferred);
        }
        if (work) {
            launch_prefetch(std::move(work));
        }
    }

    mutable std::mutex prefetch_mutex;
    mutable std::thread prefetch_thread;
    mutable std::function<void()> deferred;

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
                              int64_t begin, int64_t end, float * dst, uint8_t * raw) const {
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
            if (raw) { memcpy(raw + (size_t) s.i * row_size, s.buf.data(), row_size); }
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

    // strixllama: every row of this table is ~90 bytes at a random offset of a 28.8 GB tensor, so each one costs a 4 KB
    // page from the SSD. Read through the cache manager those peaked at ~155K pages/s - a 2K-token prompt's 31.7K rows
    // took ~200 ms with the GPU idle - while unbuffered reads of the aligned pages reach ~500K/s with 256 in flight
    // (tmp/kvr/iops_test.cpp). The handle must then be the only one: while any buffered handle on the file is open,
    // Windows keeps its cache map active and every non-cached read pays for coherency (~8K pages/s), so the reader
    // opens the file unbuffered alone (`unbuffered`, load_lazy_reader; STRIX_PLE_UNBUFFERED=0 keeps buffered reads).
    static constexpr size_t UNBUF_PAGE = 4096;
    static constexpr int    UNBUF_WORKERS = 16;
    bool unbuffered = false;

    // strixllama: the drive and its link drop to a low-power state within ~1 ms of idle; the first read after that
    // takes ~640 us instead of ~150 (tmp/dec041/ssd_latency.cpp). A decode step's gather nearly always misses a row and
    // starts ~0.5-1.5 ms after the GPU step ends, with the GPU idle: wake() reads one page ahead of it, so its reads find
    // the drive up. Fire and forget; a wake still in flight is left alone. STRIX_PLE_WAKE=0: off
    mutable std::mutex wake_mutex;
    mutable OVERLAPPED wake_ov{};
    mutable HANDLE     wake_event   = nullptr;
    mutable uint8_t *  wake_buf     = nullptr;
    mutable bool       wake_pending = false;

    void wake() const {
        static const bool off = getenv("STRIX_PLE_WAKE") && atoi(getenv("STRIX_PLE_WAKE")) == 0;
        if (off || !unbuffered) {
            return;
        }
        std::lock_guard<std::mutex> lock(wake_mutex);
        if (wake_pending) {
            if (!HasOverlappedIoCompleted(&wake_ov)) {
                return;
            }
            DWORD got = 0;
            GetOverlappedResult(fd, &wake_ov, &got, FALSE);
            wake_pending = false;
        }
        if (!wake_buf) {
            wake_buf   = (uint8_t *) _aligned_malloc(UNBUF_PAGE, UNBUF_PAGE);
            wake_event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!wake_buf || !wake_event) {
                return;
            }
        }
        const uint64_t p0 = (uint64_t) base / UNBUF_PAGE * UNBUF_PAGE;
        wake_ov = {};
        wake_ov.Offset     = (DWORD) p0;
        wake_ov.OffsetHigh = (DWORD) (p0 >> 32);
        wake_ov.hEvent     = wake_event;
        ResetEvent(wake_event);
        if (ReadFile(fd, wake_buf, (DWORD) UNBUF_PAGE, nullptr, &wake_ov) || GetLastError() == ERROR_IO_PENDING) {
            wake_pending = true;
        }
    }

    void run_range_unbuffered(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                              int64_t begin, int64_t end, float * dst, uint8_t * raw) const {
        struct slot {
            io_context io;
            OVERLAPPED ov{};
            uint8_t * buf = nullptr;
            uint64_t page0 = 0;
            DWORD len = 0;
            int64_t i = -1, j = -1;   // pairs[i..j] are served by this read
            bool pending = false;
            ~slot() { if (buf) { _aligned_free(buf); } }
        };
        const int n_slots = (int) std::min<int64_t>(STRIX_IO_DEPTH, std::max<int64_t>(1, end - begin));
        std::vector<slot> slots((size_t) n_slots);
        std::vector<HANDLE> events((size_t) n_slots);
        for (int k = 0; k < n_slots; ++k) {
            slots[k].buf = (uint8_t *) _aligned_malloc(2 * UNBUF_PAGE, UNBUF_PAGE);
            if (!slots[k].buf) {
                throw std::bad_alloc();
            }
            events[k] = slots[k].io.event;
        }
        auto offset_of = [&](int32_t row) { return (uint64_t) base + (uint64_t) row * row_size; };
        int64_t next = begin;
        int live = 0;

        auto issue = [&](slot & s) {
            const int64_t i = next;
            const uint64_t p0 = offset_of(pairs[i].first) / UNBUF_PAGE * UNBUF_PAGE;
            int64_t j = i;   // every following row that ends within the two pages from p0 comes with this read
            while (j + 1 < end && offset_of(pairs[j + 1].first) + row_size <= p0 + 2 * UNBUF_PAGE) { ++j; }
            s.len = (DWORD) (offset_of(pairs[j].first) + row_size - p0 <= UNBUF_PAGE ? UNBUF_PAGE : 2 * UNBUF_PAGE);
            s.i = i; s.j = j; s.page0 = p0; next = j + 1;
            s.ov = {};
            s.ov.Offset     = (DWORD) p0;
            s.ov.OffsetHigh = (DWORD) (p0 >> 32);
            s.ov.hEvent     = s.io.event;
            ResetEvent(s.io.event);
            if (!ReadFile(fd, s.buf, s.len, nullptr, &s.ov) && GetLastError() != ERROR_IO_PENDING) {
                throw std::runtime_error(format("lazy unbuffered read at %llu failed: %lu", (unsigned long long) p0, (unsigned long) GetLastError()));
            }
            s.pending = true; ++live;
        };

        auto finish = [&](slot & s) {
            DWORD got = 0;
            if (!GetOverlappedResult(fd, &s.ov, &got, TRUE)) {
                throw std::runtime_error(format("lazy unbuffered read completion failed: %lu", (unsigned long) GetLastError()));
            }
            for (int64_t k = s.i; k <= s.j; ) {
                int64_t kk = k;
                while (kk + 1 <= s.j && pairs[kk + 1].first == pairs[k].first) { ++kk; }
                const uint64_t o = offset_of(pairs[k].first) - s.page0;
                if (o + row_size > got) {
                    throw std::runtime_error(format("lazy unbuffered read at %llu came back short", (unsigned long long) s.page0));
                }
                if (raw) { memcpy(raw + (size_t) k * row_size, s.buf + o, row_size); }
                float * first = dst + (size_t) pairs[k].second * head_dim;
                if (to_float) {
                    to_float(s.buf + o, first, head_dim);
                } else {
                    memcpy(first, s.buf + o, (size_t) head_dim * sizeof(float));
                }
                for (int64_t m = k + 1; m <= kk; ++m) {
                    memcpy(dst + (size_t) pairs[m].second * head_dim, first, (size_t) head_dim * sizeof(float));
                }
                k = kk + 1;
            }
            s.pending = false; --live;
        };

        for (auto & s : slots) { if (next >= end) { break; } issue(s); }
        // whichever read completes first is finished and replaced, so all slots stay in flight
        while (live > 0) {
            const DWORD r = WaitForMultipleObjects((DWORD) n_slots, events.data(), FALSE, INFINITE);
            if (r >= WAIT_OBJECT_0 + (DWORD) n_slots) {
                throw std::runtime_error(format("lazy unbuffered wait failed: %lu", (unsigned long) GetLastError()));
            }
            slot & s = slots[r - WAIT_OBJECT_0];
            if (!s.pending) {   // an idle slot's event stays set from its last read
                ResetEvent(s.io.event);
                continue;
            }
            finish(s);
            if (next < end) {
                issue(s);
            } else {
                ResetEvent(s.io.event);
            }
        }
    }

#else
    using file_handle = int;
    void wake() const {}
#endif
    llama_lazy_reader(file_handle fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim, bool unbuffered = false)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
#ifdef _WIN32
        this->unbuffered = unbuffered && row_size <= UNBUF_PAGE;
#else
        (void) unbuffered;
#endif
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    // strixllama: reads bypass the file cache (the model then must not keep a mapping of the file, see load_tensors)
    bool is_unbuffered() const {
#ifdef _WIN32
        return unbuffered;
#else
        return false;
#endif
    }

    ~llama_lazy_reader() {
        if (now_thread.joinable()) {
            now_thread.join();
        }
#ifdef _WIN32
        if (prefetch_thread.joinable()) {
            prefetch_thread.join();
        }
        // a wake-up read still in flight writes its buffer until it completes
        if (wake_pending) {
            DWORD got = 0;
            GetOverlappedResult(fd, &wake_ov, &got, TRUE);
        }
        if (wake_buf) {
            _aligned_free(wake_buf);
        }
        if (wake_event) {
            CloseHandle(wake_event);
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
        const auto strixllama_t1 = std::chrono::steady_clock::now();
        auto strixllama_t2 = strixllama_t1, strixllama_t3 = strixllama_t1;

        int64_t n_hit = 0;
        if (!cache_on()) {
            read_pairs(pairs, dst, nullptr);
        } else {
            // strixllama: rows found in the cache are copied out under the lock and dequantized after it; the rest are
            // read as before, their raw bytes kept, and put in the cache
            std::vector<std::pair<int32_t, int32_t>> miss;
            std::vector<int64_t> hit;          // pairs index of a cached row's first occurrence
            std::vector<uint8_t> hit_raw;      // its raw bytes, in the order of hit
            miss.reserve(pairs.size());
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache_alloc();
                for (int64_t i = 0; i < n; ) {
                    int64_t j = i;
                    while (j + 1 < n && pairs[j + 1].first == pairs[i].first) { ++j; }
                    const uint8_t * c = cache_find(pairs[i].first);
                    if (c) {
                        hit.push_back(i);
                        hit_raw.insert(hit_raw.end(), c, c + row_size);
                        n_hit += j - i + 1;
                    } else {
                        for (int64_t k = i; k <= j; ++k) { miss.push_back(pairs[k]); }
                    }
                    i = j + 1;
                }
            }
            for (size_t h = 0; h < hit.size(); ++h) {
                const int64_t i = hit[h];
                float * first = dst + (size_t) pairs[i].second * head_dim;
                const uint8_t * src = hit_raw.data() + h * row_size;
                if (to_float) {
                    to_float(src, first, head_dim);
                } else {
                    memcpy(first, src, (size_t) head_dim * sizeof(float));
                }
                for (int64_t k = i + 1; k < n && pairs[k].first == pairs[i].first; ++k) {
                    memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
                }
            }
            strixllama_t2 = std::chrono::steady_clock::now();
            if (!miss.empty()) {
                std::vector<uint8_t> raw(miss.size() * row_size);
                read_pairs(miss, dst, raw.data());
                strixllama_t3 = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(cache_mutex);
                for (size_t k = 0; k < miss.size(); ++k) {
                    if (k == 0 || miss[k].first != miss[k - 1].first) {
                        cache_put(miss[k].first, raw.data() + k * row_size);
                    }
                }
            }
        }

        if (strixllama_trace) {
            const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - strixllama_t0).count();
            int64_t uniq = n ? 1 : 0;
            for (int64_t i = 1; i < n; ++i) {
                if (pairs[i].first != pairs[i - 1].first) { ++uniq; }
            }
            auto ms_of = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "PLE_GATHER rows=%lld uniq=%lld cached=%lld row_bytes=%zu ms=%.1f (sort %.1f hits %.1f read %.1f)\n",
                    (long long) n, (long long) uniq, (long long) n_hit, row_size, ms, ms_of(strixllama_t0, strixllama_t1),
                    ms_of(strixllama_t1, strixllama_t2), ms_of(strixllama_t2, strixllama_t3));
        }
    }

    // strixllama: a cache of raw rows in RAM (STRIX_PLE_CACHE_MB, default 400; 0 = none). Reads bypass the file cache
    // (unbuffered), so without it a row is read from the SSD every time: a 2K-token prompt's n-gram rows cost ~200 ms
    // with the GPU idle, a repeated prompt or a common phrase as much as a new one. Direct-mapped by row: a slot holds
    // its row's index and raw bytes.
    mutable std::mutex cache_mutex;
    mutable std::vector<uint8_t> cache_mem;
    mutable size_t cache_slots = 0;     // a power of 2
    size_t cache_stride() const { return (sizeof(int32_t) + row_size + 15) / 16 * 16; }
    static long cache_mb() {
        static const long mb = getenv("STRIX_PLE_CACHE_MB") ? atol(getenv("STRIX_PLE_CACHE_MB")) : 400;
        return mb;
    }
    bool cache_on() const { return cache_mb() > 0; }
    void cache_alloc() const {
        if (cache_slots) { return; }
        size_t slots = 1;
        while (slots * 2 * cache_stride() <= (size_t) cache_mb() << 20) { slots *= 2; }
        cache_mem.assign(slots * cache_stride(), 0);
        for (size_t k = 0; k < slots; ++k) { *(int32_t *) (cache_mem.data() + k * cache_stride()) = -1; }
        cache_slots = slots;
    }
    uint8_t * cache_slot(int32_t row) const {
        const size_t h = (size_t) ((uint32_t) row * 2654435761u) & (cache_slots - 1);
        return cache_mem.data() + h * cache_stride();
    }
    const uint8_t * cache_find(int32_t row) const {
        const uint8_t * s = cache_slot(row);
        return *(const int32_t *) s == row ? s + sizeof(int32_t) : nullptr;
    }
    void cache_put(int32_t row, const uint8_t * data) const {
        uint8_t * s = cache_slot(row);
        *(int32_t *) s = row;
        memcpy(s + sizeof(int32_t), data, row_size);
    }

    // the sorted pairs' rows read by the worker threads into dst (and their raw bytes into raw, by pairs index)
    void read_pairs(const std::vector<std::pair<int32_t, int32_t>> & pairs, float * dst, uint8_t * raw) const {
        const int64_t n = (int64_t) pairs.size();
        // small gathers are not worth a thread per row
#ifdef _WIN32
        const int n_workers = (int) std::min<int64_t>(unbuffered ? std::min(n_threads, UNBUF_WORKERS) : n_threads,
                                                      std::max<int64_t>(1, n / 32));
#else
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, n / 32));
#endif
        // a worker's range must not split a row's run of pairs: each run is read, and captured, once
        std::vector<int64_t> cut((size_t) n_workers + 1);
        for (int w = 0; w <= n_workers; ++w) {
            int64_t c = n * w / n_workers;
            while (c > 0 && c < n && pairs[c].first == pairs[c - 1].first) { ++c; }
            cut[(size_t) w] = c;
        }

        auto run_chunk = [&](int w, std::exception_ptr & err) {
            try {
                if (cut[(size_t) w] < cut[(size_t) w + 1]) {
                    run_range(pairs, cut[(size_t) w], cut[(size_t) w + 1], dst, raw);
                }
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
    }

    // strixllama: rows gathered ahead of time for a batch a caller announced (llama_strix_prefetch). The gather for
    // that batch takes them instead of reading them again, so it costs a copy rather than ~0.2-0.3 s of reads with
    // the GPU idle. Two slots: the next batch's pregather can finish before this batch takes its own.
    struct pregathered {
        std::vector<int32_t> rows;
        std::vector<uint8_t> data;      // rows.size() * head_dim floats
        uint64_t id    = 0;             // strixllama: a pregather announced before it ran (pregather_now) has one ...
        bool     ready = true;          // ... and is not ready while it runs: a take of its rows waits for it
    };
    mutable std::mutex pre_mutex;
    mutable std::condition_variable pre_cv;
    mutable uint64_t pre_next_id = 1;
    mutable std::vector<pregathered> pre;

    // keep the newest few: a running pregather is never dropped, its taker waits for it
    void pre_trim() const {
        while (pre.size() > 3) {
            auto it = std::find_if(pre.begin(), pre.end(), [](const pregathered & p) { return p.ready; });
            if (it == pre.end()) {
                break;
            }
            pre.erase(it);
        }
    }

    void pregather(const int32_t * rows, int64_t n, uint64_t id = 0) const {
        std::vector<uint8_t> data((size_t) n * head_dim * sizeof(float));
        bool ok = true;
        try {
            gather(rows, n, (float *) data.data());
        } catch (...) {
            ok = false;   // an announced entry is still released below: its taker reads the rows itself
            if (!id) {
                throw;
            }
        }
        {
            std::lock_guard<std::mutex> lock(pre_mutex);
            auto it = id ? std::find_if(pre.begin(), pre.end(), [id](const pregathered & p) { return p.id == id; }) : pre.end();
            if (it != pre.end()) {
                if (ok) {
                    it->data.swap(data);
                } else {
                    pre.erase(it);
                }
            } else if (!id) {
                pregathered p;
                p.rows.assign(rows, rows + n);
                p.data.swap(data);
                pre.push_back(std::move(p));
            }
            for (auto & p : pre) {
                if (p.id == id) {
                    p.ready = true;
                }
            }
            pre_trim();
        }
        pre_cv.notify_all();
    }

    // strixllama: a pregather of the batch about to run - its first chunk, which no earlier batch announced - started at
    // once on a thread of its own, so its reads overlap the batch's preparation and graph build; announced first, so the
    // batch's gather waits for it rather than reading the rows again
    void pregather_now(std::vector<int32_t> rows) const {
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(pre_mutex);
            for (const auto & p : pre) {     // already announced or gathered
                if (p.rows.size() >= rows.size() && std::equal(rows.begin(), rows.end(), p.rows.begin())) {
                    return;
                }
            }
            id = pre_next_id++;
            pregathered p;
            p.rows  = rows;
            p.id    = id;
            p.ready = false;
            pre.push_back(std::move(p));
        }
        std::lock_guard<std::mutex> lock(now_mutex);
        std::thread prev = std::move(now_thread);
        now_thread = std::thread([this, prev = std::move(prev), rows = std::move(rows), id]() mutable {
            if (prev.joinable()) {
                prev.join();
            }
            try {
                pregather(rows.data(), (int64_t) rows.size(), id);
            } catch (...) {
                LLAMA_LOG_WARN("lazy pregather skipped after an exception\n");
            }
        });
    }
    mutable std::mutex  now_mutex;
    mutable std::thread now_thread;

    // the gathered rows for `rows`, when a pregather covered them: the batch a caller announced can come out shorter
    // (it stops at a checkpoint or a message boundary), so a pregather that starts with these rows serves too
    bool take_pregathered(const int32_t * rows, int64_t n, std::vector<uint8_t> & out) const {
        std::unique_lock<std::mutex> lock(pre_mutex);
        // a match still being gathered: wait for it (it is released whether or not its reads succeed)
        auto wait_ready = [&](uint64_t id) {
            pre_cv.wait(lock, [&]() {
                auto it = std::find_if(pre.begin(), pre.end(), [id](const pregathered & p) { return p.id == id; });
                return it == pre.end() || it->ready;
            });
            return std::find_if(pre.begin(), pre.end(), [id](const pregathered & p) { return p.id == id; });
        };
        for (auto it = pre.begin(); it != pre.end(); ++it) {
            if ((int64_t) it->rows.size() < n || memcmp(it->rows.data(), rows, (size_t) n * sizeof(int32_t)) != 0) {
                continue;
            }
            if (!it->ready) {
                it = wait_ready(it->id);
                if (it == pre.end()) {
                    return false;
                }
            }
            const size_t bytes = (size_t) n * head_dim * sizeof(float);
            if (it->data.size() == bytes) {
                out.swap(it->data);
            } else {
                out.resize(bytes);
                memcpy(out.data(), it->data.data(), bytes);
            }
            pre.erase(it);
            return true;
        }
        // strixllama: a prompt chunk that shares its pass with other conversations sits inside the batch, after their
        // tokens (or its pregather ran short of it): take the longest run of pregathered rows found in the batch and
        // read the rows around it. A row's data does not depend on where it sits, so any match is exact.
        int64_t best = -1, best_o = 0, best_len = 0;
        for (size_t k = 0; k < pre.size(); ++k) {
            const auto & e = pre[k];
            if (e.rows.empty()) { continue; }
            for (int64_t o = 0; o < n && best_len < n - o; ++o) {
                if (rows[o] != e.rows[0]) { continue; }
                const int64_t lim = std::min<int64_t>((int64_t) e.rows.size(), n - o);
                int64_t len = 1;
                while (len < lim && rows[o + len] == e.rows[len]) { ++len; }
                if (len > best_len) { best = (int64_t) k; best_o = o; best_len = len; }
            }
        }
        // a short coincidental match must not use up the pregather of a later batch
        if (best < 0 || (best_len < 1024 && best_len < (int64_t) pre[best].rows.size())) {
            return false;
        }
        if (!pre[best].ready) {
            const auto it = wait_ready(pre[best].id);
            if (it == pre.end()) {
                return false;
            }
            best = it - pre.begin();
        }
        const size_t rb = head_dim * sizeof(float);
        out.resize((size_t) n * rb);
        memcpy(out.data() + (size_t) best_o * rb, pre[best].data.data(), (size_t) best_len * rb);
        pre.erase(pre.begin() + best);
        lock.unlock();
        if (best_o > 0) {
            gather(rows, best_o, (float *) out.data());
        }
        if (best_o + best_len < n) {
            gather(rows + best_o + best_len, n - best_o - best_len, (float *) (out.data() + (size_t) (best_o + best_len) * rb));
        }
        return true;
    }

    // populate the page cache for the rows a later gather() will read; never writes any output, so a wrong
    // prediction only wastes readahead. Safe to call concurrently with gather().
    void prefetch(const int32_t * rows, int64_t n) const {
#ifdef _WIN32
        if (unbuffered) {
            return;
        }
#endif
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
                   int64_t begin, int64_t end, float * dst, uint8_t * raw = nullptr) const {
#ifdef _WIN32
        if (unbuffered) {
            run_range_unbuffered(pairs, begin, end, dst, raw);
        } else {
            run_range_overlapped(pairs, begin, end, dst, raw);
        }
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
            if (raw) { memcpy(raw + (size_t) i * row_size, bounce.data(), row_size); }
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
