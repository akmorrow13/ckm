from experiments import *
import pickle
import yaml

base_exp =  load(open("./keystone_pipeline/cifar_cluster.exp"))
# 1 Layer sweep
results = []
for bw1 in [0.5, 0.7, 1, 1.2]:
    for bw in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5,0.7,2, 5 ,10, 20, 100]:
        exp_copy = base_exp.copy()
        exp_copy['bandwidth'][-1] = bw
        exp_copy['bandwidth'][0] = bw1
        fname = "/tmp/bw_sweep_{0}".format(bw)
        sweep_file = open(fname, "w+")
        yaml.dump(exp_copy, sweep_file)
        sweep_file.close()
        result = scala_run(exp_copy, fname)
        results.append(result)

for r in results:
    print tabulate(r, headers="keys")
    print "===================================="
