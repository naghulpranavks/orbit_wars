import sys, os, shutil, tarfile, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract(a, d):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    with tarfile.open(a) as t: t.extractall(d)

def run_games(td, rd, n=30, seed_base=1000):
    def game(seed, swap):
        r = subprocess.run([sys.executable,'run_game_runner.py',str(seed),td,rd,str(swap)],
            capture_output=True,text=True,encoding='utf-8',timeout=120)
        for l in r.stdout.strip().split('\n'):
            if l.startswith('RESULT:'): return l.split(':')[1]
        return 'ERROR'
    wins=losses=draws=0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(game,seed_base+i,i%2==1) for i in range(n)]
        for f in as_completed(futs):
            r=f.result()
            if r=='TEST_WIN': wins+=1
            elif r=='REF_WIN': losses+=1
            elif r=='DRAW': draws+=1
    return wins, losses, draws, n

bots = {
    'V16': 'submission_v16.tar.gz',
    'V13': 'submission_v13_flowbot.tar.gz',
    'V11': 'submission_v11.tar.gz'
}

v20d = os.path.abspath('buddy_model')
print("=== V21 vs Random High Level Bots ===")

for name, tar in bots.items():
    d = os.path.abspath(f'iso_{name}')
    extract(tar, d)
    w,l,d_res,n = run_games(v20d, d, n=30, seed_base=12000)
    print(f"  V21 vs {name}: {w}W/{l}L  ({w/n*100:.0f}%)")
    shutil.rmtree(d, ignore_errors=True)

