# Usage: python3 build.py <v2 dir> <out fragment> <out standalone>
import json, sys, os, hashlib, base64, subprocess
d = sys.argv[1]; data = json.load(open(os.path.join(d, 'data.json')))
cache = os.path.join(d, 'audio'); os.makedirs(cache, exist_ok=True)
jobs = []
def add(key, text, rate='-8%'):
    fid = hashlib.md5((key + '|' + text + '|' + rate).encode()).hexdigest()[:12]
    jobs.append({'key': key, 'id': fid, 'text': text, 'rate': rate})
for ch, nm in data['names'].items():
    add('n:' + ch, nm + '.', '-12%')
    add('t:' + ch, 'Это буква ' + nm + '!')
words = set(x[1] for x in data['letters']) | set(w.lower() for lvl in data['words'].values() for w, _ in lvl)
for w in sorted(words): add('w:' + w, w + '.', '-12%')
for k, t in data['phrases'].items(): add('p:' + k, t)
json.dump(jobs, open(os.path.join(d, 'jobs.json'), 'w'), ensure_ascii=False)
subprocess.run(['python3', os.path.join(os.path.dirname(__file__), 'tts.py'), os.path.join(d, 'jobs.json'), cache], check=True, stdout=subprocess.DEVNULL)
trim = os.path.join(d, 'audio_trim'); os.makedirs(trim, exist_ok=True)
AF = ('silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.04,'
      'areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.12,areverse')
voice = {}
for j in jobs:
    src = os.path.join(cache, j['id'] + '.mp3'); p = os.path.join(trim, j['id'] + '.mp3')
    if os.path.exists(src) and os.path.getsize(src) > 0 and not os.path.exists(p):
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', src, '-af', AF, '-ac', '1', '-ar', '24000', '-b:a', '40k', p], check=True)
    if not os.path.exists(p) or os.path.getsize(p) == 0: print('MISSING', j['key']); continue
    voice[j['key']] = base64.b64encode(open(p, 'rb').read()).decode()
tpl = open(os.path.join(d, 'game.tpl.html')).read()
frag = tpl.replace('__DATA__', json.dumps({k: data[k] for k in ('letters', 'stages', 'words', 'names')}, ensure_ascii=False)) \
          .replace('__VOICE__', json.dumps(voice))
open(sys.argv[2], 'w').write(frag)
head = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        '<meta name="theme-color" content="#2E5A4C">\n<meta name="apple-mobile-web-app-capable" content="yes">\n'
        '<style>[hidden]{display:none!important}:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n'
        '</head>\n<body>\n')
open(sys.argv[3], 'w').write(head + frag + '\n</body>\n</html>\n')
print('clips', len(voice), 'of', len(jobs), 'size KB', len(frag) // 1024)
