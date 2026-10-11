import asyncio, ssl, os, sys, json
import edge_tts, edge_tts.communicate as c
# Behind a TLS-intercepting proxy, point SSL_CERT_FILE at its CA bundle.
if os.environ.get('SSL_CERT_FILE'):
    c._SSL_CTX = ssl.create_default_context(cafile=os.environ['SSL_CERT_FILE'])
PROXY = os.environ.get('HTTPS_PROXY')
async def one(text, out, voice, rate, pitch):
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, proxy=PROXY)
    await comm.save(out)
async def main():
    jobs = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    for j in jobs:
        p = os.path.join(outdir, j['id'] + '.mp3')
        if os.path.exists(p) and os.path.getsize(p) > 0: continue
        for attempt in range(3):
            try:
                await one(j['text'], p, j.get('voice','ru-RU-SvetlanaNeural'), j.get('rate','-8%'), j.get('pitch','+0Hz')); break
            except Exception as e:
                print('retry', j['id'], e); await asyncio.sleep(2)
        print(j['id'], os.path.getsize(p) if os.path.exists(p) else 'FAIL')
asyncio.run(main())
