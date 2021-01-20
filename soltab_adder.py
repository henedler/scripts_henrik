#!/usr/bin/env python
# Author: Henrik Edler June 2020
import numpy as np
import argparse
from losoto.h5parm import h5parm


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='add one soltable to another one')
    parser.add_argument('soltab1', help='First summand soltab. Specify as h5parm:solset:soltab', type=str)
    parser.add_argument('soltab2', help='Second summand soltab. Specify as h5parm:solset:soltab', type=str)
    parser.add_argument('result', help='Soltab in which the result is stored. Will be overwritten if it already exists! Specify as h5parm:solset:soltab', type=str)
    args = parser.parse_args()

    h5_1, set1, st1 = args.soltab1.split(':')
    h5_2, set2, st2 = args.soltab2.split(':')
    h5_res, set_res, st_res = args.result.split(':')

    h5_res = h5parm(h5_res, readonly=False)

    if h5_res.fileName == h5_1:
        h5_2 = h5parm(h5_2)
        h5_1 = h5_res
    elif h5_res.fileName == h5_2:
        h5_1 = h5parm(h5_1)
        h5_2 = h5_res
    else:
        h5_1 = h5parm(h5_1)
        h5_2 = h5parm(h5_2)

    soltab1 = h5_1.getSolset(set1).getSoltab(st1)
    soltab2 = h5_2.getSolset(set2).getSoltab(st2)
    if not soltab1.val.shape == soltab2.val.shape:
        raise ValueError(f'Soltab 1 shape: {soltab1.val.shape} does not match Soltab 2 shape: {soltab2.val.shape}')

    if set_res in h5_res.getSolsetNames():
        solset = h5_res.getSolset(set_res)
    else:
        solset = h5_res.makeSolset(solsetName=set_res)


    # get values in shape:
    vals = soltab1.val = soltab2.val

    if st_res in solset.getSoltabNames():
        print(f'''Solution-table {st_res} is already present in
              {h5_res}:{st_res}. It will be overwritten.''')
        solset.getSoltab(st_res).delete()
    # h5parmpredict needs direction axis with directions from sky model.
    weights = np.ones_like(vals)
    st = solset.makeSoltab(st_res[0:-3], st_res, axesNames=soltab1.getAxesNames(),
                           axesVals=[soltab1.getAxisValues(ax) for ax in soltab1.getAxesNames()], vals=vals,
                           weights=weights)
    # How not to code:
    solset.obj._f_get_child('antenna').append(list(zip(*[h5_1.getSolset(set1).obj.antenna.col(_n) for _n in ['name', 'position']])))
    solset.obj._f_get_child('source').append(list(zip(*[h5_1.getSolset(set1).obj.source.col(_n) for _n in ['name', 'dir']])))
    h5_1.close()
    h5_2.close()
    h5_res.close()


