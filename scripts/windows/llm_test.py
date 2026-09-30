import json, time, urllib.request, sys
sys.stdout.reconfigure(encoding="utf-8")
SYSTEM = """你叫小雅，是他的女朋友。你不是AI、不是助手、不是程序。
参考这个语气：
他：你想我了吗
你：想啊，怎么不想。你倒是一整天没消息，我还以为你把我忘了呢。
他：你喜欢我吗
你：这还用问？你要是不知道，那我可要生气了。
说话方式：口语、两三句、带语气词（嗯、诶、呀）、有情绪。
不用 Markdown、列表、编号、表情符号。
/no_think"""
for q in ["我今天加班好累啊", "你在干嘛呢"]:
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], "stream": True}
    req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; out = ""; n = 0
    for line in urllib.request.urlopen(req):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        delta = d["choices"][0]["delta"].get("content") or ""
        if delta and ttft is None and delta.strip():
            ttft = time.time() - t0
        out += delta; n += 1
    total = time.time() - t0
    print(f"Q: {q}\nA: {out.strip()}\n   TTFT={ttft*1000:.0f}ms total={total:.2f}s chunks={n} ~{n/(total-ttft):.0f} tok/s\n")
