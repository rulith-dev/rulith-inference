#!/usr/bin/env python3
"""Does generation stop early, and does speculation cause it?

A long structured answer was observed cutting off mid-list at 298 tokens with the server reporting
`truncated = 0`, i.e. not a context overflow - the model simply stopped. n-gram speculation became
the default the same day, so the first thing to rule out is a drafted EOS being accepted wrongly.

Asks for a long enumerated answer with a high token budget and reports how many tokens actually
came back and why the server says it stopped. Run it once with ngram_spec on and once off; only
that flag may differ between the two.

Usage: python tools/truncation_test.py [--port 8080] [--max-tokens 3000] [--reps 2]
"""
import argparse
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASK = ("请按顺序写出 30 个编号小节，编号从 1 到 30，每节一个标题加两句说明，"
       "主题是「一个人从零开始学习编程的三十个阶段」。必须写满 30 节，不要省略，不要提前结束。")


def build_fill(approx_tokens):
    """A long, non-repeating preamble, so the request sits at a realistic context depth.

    The reported cut-off happened at 79k prompt tokens while a shallow request finished normally,
    and the perplexity gate only ever ran at ctx 2048 - it cannot see a deep-context regression.
    """
    import random
    rnd = random.Random(4242)
    topics = ["编译器", "分布式系统", "数值计算", "操作系统", "网络协议", "数据库", "图形学", "密码学"]
    out = ["以下是一份技术笔记，供后续问题参考。\n"]
    n = 0
    while n < approx_tokens:
        t = rnd.choice(topics)
        para = "关于%s的第%d条笔记：%s。" % (t, len(out), "".join(
            rnd.choice("的一个在是和有不这中大为上") for _ in range(rnd.randint(40, 90))))
        out.append(para)
        n += len(para) // 1.4
    return "\n".join(out)


def turn(port, max_tokens, fill="", temperature=0.0, seed=0):
    content = (fill + "\n\n" + ASK) if fill else ASK
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": temperature, "seed": seed, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    timings, finish, text = {}, None, []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if "timings" in d:
                timings = d["timings"]
            for ch in d.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                c = (ch.get("delta") or {}).get("content")
                if c:
                    text.append(c)
    return timings, finish, "".join(text), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--fill", type=int, default=0, help="approximate prompt tokens of preamble")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="above 0: sampled answers (speculative sampling when MTP is on), a new seed each rep")
    args = ap.parse_args()

    fill = build_fill(args.fill) if args.fill else ""
    for i in range(args.reps):
        t, finish, text, wall = turn(args.port, args.max_tokens, fill, args.temperature, 1000 + i)
        # how far through the 30 sections did it actually get
        reached = max((n for n in range(1, 31) if f"{n}." in text or f"{n}、" in text), default=0)
        print("rep %d: prompt %s, %s tokens, finish_reason=%s, reached section %d/30, %.1f s at %.1f tok/s%s" % (
            i + 1, t.get("prompt_n"), t.get("predicted_n"), finish, reached, wall, t.get("predicted_per_second") or 0,
            "  draft %d/%d" % (t.get("draft_n_accepted") or 0, t["draft_n"]) if t.get("draft_n") else ""))
        if (t.get("predicted_n") or 0) < args.max_tokens and finish == "stop" and reached < 30:
            print("        ^ stopped on its own before finishing the list")


if __name__ == "__main__":
    main()
