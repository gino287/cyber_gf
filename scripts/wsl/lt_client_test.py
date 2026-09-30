"""Headless stand-in for LiveTalking index.html: connect via WebRTC, send text, measure."""
import asyncio, json, sys, time
import aiohttp, numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
TEXT = "你好呀，我是小雅，今天过得怎么样？"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lt_frames"


async def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")
    stats = {"v": [], "a": []}
    t0 = time.time()

    @pc.on("track")
    def on_track(track):
        async def pump():
            n = 0
            while True:
                try:
                    f = await track.recv()
                except Exception:
                    return
                t = time.time() - t0
                if track.kind == "video":
                    stats["v"].append(t)
                    n += 1
                    if n % 25 == 0:
                        f.to_image().save(f"{OUT}/v_{t:06.2f}.jpg")
                else:
                    pcm = f.to_ndarray().astype(np.float32)
                    stats["a"].append((t, float(np.sqrt((pcm ** 2).mean()))))
        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())
    async with aiohttp.ClientSession() as s:
        r = await s.post(f"{BASE}/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        ans = json.loads(await r.text())
        print("offer ->", {k: v for k, v in ans.items() if k != "sdp"})
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        await asyncio.sleep(6)
        t_send = time.time() - t0
        r = await s.post(f"{BASE}/human", json={"text": TEXT, "type": "echo", "interrupt": True, "sessionid": str(ans["sessionid"])})
        print("human ->", await r.text(), "at", round(t_send, 2))
        await asyncio.sleep(12)
    await pc.close()

    v = stats["v"]
    print(f"video frames={len(v)} span={v[-1]-v[0]:.1f}s fps={len(v)/(v[-1]-v[0]):.1f}" if len(v) > 1 else "NO VIDEO")
    loud = [t for t, e in stats["a"] if e > 300 and t > t_send]
    if loud:
        print(f"audio: first voiced frame {loud[0]-t_send:.2f}s after /human, voiced span {loud[-1]-loud[0]:.1f}s")
    else:
        print("audio: no voiced frames", len(stats["a"]))


asyncio.run(main())
