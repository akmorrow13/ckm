from experiments import *
import pickle
import yaml

base_exp =  load(open("./keystone_pipeline/cifar_cluster.exp"))
base_exp['expid'] = "4x4x128_gamma_reg_sweep"
# 1 Layer sweep
results = []
for gamma in [1e-6, 5e-6, 1e-7]:
    for reg in [1e-3, 5e-3, 1e-4]:
        exp_copy = base_exp.copy()
        exp_copy["kernelGamma"] = gamma
        fname = "/tmp/gamma_reg_sweep_{0}_{1}".format(gamma,  reg)
        sweep_file = open(fname, "w+")
        yaml.dump(exp_copy, sweep_file)
        sweep_file.close()
        result = scala_run(exp_copy, fname)
        results.append(result)

for i,r in enumerate(results):
    if (i == 0):
        header = True
    else:
        header = False
    print r.to_csv(header=header)
