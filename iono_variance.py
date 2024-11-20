from losoto.h5parm import h5parm
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import glob

statpos = pd.read_csv('station_pos_LBA', sep='\t', names=['ant','x','y','z'])
statpos['x'] = statpos['x']-3826577.462
statpos['y'] = statpos['y']-461022.624
statpos['z'] = statpos['z']-5064892.526

statpos['d'] = np.sqrt((statpos['x'])**2 + (statpos['y'])**2 + (statpos['z'])**2)
statpos['ant'][0]

statpos[['ant','d']].to_csv('LBA_dist.csv')

files = glob.glob('h5parms/*')
#t0 = pd.to_datetime('2022-04-01 00:00:00.00',format='%Y-%m-%d %H:%M:%S.%f')


def get_row(path):
    h5 = h5parm(path)
    ss = h5.getSolset('sol000')

    if 'tec000' not in ss.getSoltabNames():
        print(f'No tec000 in {path}')
        return None

    st = ss.getSoltab('tec000')  # , sel={'ant':'RS503LBA'}
    t = st.getAxisValues('time')

    ant = st.getAxisValues('ant')

    t = pd.to_datetime(t / (24 * 60 * 60) + 2400000.5, unit='D', origin='julian')
    # if t[0] < t0:
    #     return None
    # else:
    #     print(path)

    data = np.squeeze(st.getValues()[0])
    df = pd.DataFrame(data, index=t, columns=st.getAxisValues('ant'))
    df_out = pd.DataFrame(columns=['t', 'id', *statpos['ant'].values])
    antdat = np.nan * np.ones_like(statpos['ant'].values)

    for i, ant in enumerate(statpos['ant'].values):
        if ant in df.columns:
            antdat[statpos['ant'].values == ant] = df.std()[ant]

    df_out.loc[len(df_out.index)] = [t.mean(), path.split('/')[-1], *antdat]

    plt.scatter(statpos['ant'], antdat)
    # plt.xscale('log')
    # plt.yscale('log')

    return df_out

dfs = []
i = 0
for file in files:
    try:
        dfs.append(get_row(file))
    except:
        print(file, 'error')
        continue
    i = i+1
    print(file)
    # if i > 100:
    #     break

df_concat = pd.concat(dfs)
df_concat = df_concat.set_index('t')
df_concat['id'][np.argsort(df_concat['RS210LBA'])]
df_concat.to_csv('iono_data_all.csv')