import subprocess
import argparse

CMD_BASE = "hadoop fs -copyToLocal"

''' Script to experiment output files from hdfs to dst '''

def copy_from_hadoop(src, dst):
        p = subprocess.Popen(" ".join([CMD_BASE, src, dst]), shell=True, executable='/bin/bash')
        p.wait()
        if p.returncode != 0:
            raise Exception("invocation terminated with non-zero exit status")

def copy_train_test_from_hadoop(expid, root, dst):
    train_name = expid + "_train_features"
    test_name = expid + "_test_features"
    copy_from_hadoop(root + train_name, dst)
    copy_from_hadoop(root + test_name, dst)

def main():
    parser = argparse.ArgumentParser(description='Copy train and test files from hadoop')
    parser.add_argument('expid', help='expid  to copy')
    parser.add_argument('--root', default="/",  dest="root", help='root hdfs dir')
    parser.add_argument('--dst', default=".",  dest="dst", help='place to copy to')
    args = parser.parse_args()
    copy_train_test_from_hadoop(args.expid, args.root, args.dst)


if __name__ == "__main__":
    main()



