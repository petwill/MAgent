import pandas as pd
def go(fname):
    df = pd.read_csv(fname)
    print(df['agent'].mean()/4, df['agent_strong'].mean())

def go2(fname, a, b):
    df = pd.read_csv(fname)
    print(df['agent'].mean()/a, df['agent_strong'].mean()/b)
