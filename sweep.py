from experiments import *
import pickle

base_exp =  load(open("sample_python.exp"))
# 1 Layer sweep
for numFeatures in [50, 100, 200]:
    for reg in [0.01, 0.001]:
        for center in [True, False]:
            for weight in [1.414, 1.0, 3]:
                exp = base_exp.copy()
                exp["filters"] = [numFeatures]
                exp["reg"] = reg
                exp["center"] = center
                exp["weight"] = weight
                exp["verbose"] = True
                print "Experiment Parameters:"
                print exp
                results = python_run(exp)
                print tabulate(results, headers="keys")
                print "==========================================="



