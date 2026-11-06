import sys, os, shutil, tarfile, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

def extract(a, d):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    with tarfile.open(a) as t: t.extractall(d)
    
    # Check if main.py is in a subdirectory (e.g. tar extracted a folder)
    if not os.path.exists(os.path.join(d, 'main.py')):
        for root, dirs, files in os.walk(d):
            if 'main.py' in files:
                for f in files:
                    try:
                        shutil.move(os.path.join(root, f), os.path.join(d, f))
                    except:
                        pass
                for child_dir in dirs:
                    try:
                        shutil.move(os.path.join(root, child_dir), os.path.join(d, child_dir))
                    except:
                        pass
                break

def run_games(td, rd, n=10, seed_base=20000):
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

def main():
    bots = glob.glob('submission_v*.tar.gz')
    # Filter out any non-standard ones, take latest 10
    bots = [b for b in bots if 'numpy' not in b]
    # sort by version number
    def get_v(name):
        try:
            return int(name.replace('submission_v', '').replace('_flowbot.tar.gz', '').replace('.tar.gz', ''))
        except:
            return 0
    bots.sort(key=get_v)
    bots = bots[-10:] # Last 10 models

    v20d = os.path.abspath('buddy_model')
    print("=== V20 vs 10 Models (10 games each) ===")

    total_w = total_l = total_d = 0
    for tar in bots:
        name = tar.replace('submission_', '').replace('.tar.gz', '')
        d = os.path.abspath(f'iso_{name}')
        try:
            extract(tar, d)
            w,l,d_res,n = run_games(v20d, d, n=10, seed_base=20000)
            winrate = (w/n)*100 if n > 0 else 0
            print(f"  V20 vs {name}: {w}W/{l}L  ({winrate:.0f}%)")
            total_w += w
            total_l += l
            total_d += d_res
        except Exception as e:
            print(f"  V20 vs {name}: ERROR ({e})")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    total_games = total_w + total_l + total_d
    print(f"\nOVERALL: {total_w}W / {total_l}L / {total_d}D  ({(total_w/total_games)*100 if total_games > 0 else 0:.1f}%)")

if __name__ == '__main__':
    main()
