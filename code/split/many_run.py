#!/usr/bin/python3
from collections import defaultdict
import multiprocessing as mp
from TraceGen import *
from LambdaScheduler import LambdaScheduler
import pickle
import argparse
import os

def load_trace(num_functions, char, trace_path):
    fname = "{}-{}.pckl".format(num_functions, char)
    with open(os.path.join(trace_path, fname), "r+b") as f:
        return pickle.load(f)

def compare_pols(policy, num_functions, char, mem_capacity=32000, args=None):
    # save_pth = "/data2/alfuerst/verify-test/"
    # log_dir = "/data2/alfuerst/verify-test/logs/"
    save_pth = args.savedir
    log_dir = args.logdir
    name = "{}-{}-{}-{}.pckl".format(policy, num_functions, mem_capacity, char)
    save_pth = os.path.join(save_pth, name)

    if not os.path.exists(save_pth):
        result_dict = dict()
        evdicts = dict()
        misses = dict()

        L = LambdaScheduler(policy, mem_capacity, num_functions, char, log_dir)
        lambdas, trace = load_trace(num_functions, char, args.tracedir)
        i = 0
        for d, t in trace:
            i += 1
            L.runActivation(d, t)

        L.MemUsageHist.flush()
        L.PerformanceLog.flush()
        L.PureCacheHist.flush()

        L.MemUsageHist.close()
        L.PerformanceLog.close()
        L.PureCacheHist.close()

        # policy = string
        # L.evdict:    dict[func_name] = eviction_count
        # L.miss_stats: dict of function names whos entries are  {"misses":count, "hits":count}
        # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
        # capacity_misses: dict[func_name] = invocations_not_handled
        # len_trace: long
        data = (policy, L.evdict, L.miss_stats(), lambdas, L.capacity_misses, len(trace))
        # write-then-rename: a run killed mid-save never leaves a truncated result
        # (which later sweeps would silently skip as "already done")
        with open(save_pth + ".tmp", "w+b") as f:
            pickle.dump(data, f)
        os.replace(save_pth + ".tmp", save_pth)

        if args.delete_logs:
            # per-invocation CSV logs are ~60MB/run on the full-day trace; everything the
            # analysis needs is already in the result pickle (miss_stats read them above)
            for fname in (L.PerfLogFName, L.MemUsageFname, L.PureCacheFname):
                os.remove(fname)

    print("done", name)

def run_multiple_expts(args):
    policies = args.policies
    mems = [i for i in range(args.memmin, args.memmax, args.memstep)]
    mems = set(mems)
    results = []
    print(len(policies) * len(mems))
    with mp.Pool(args.workers) as pool:
        for policy in policies:
            for mem in mems:
                for num_func in [args.numfuncs]:
                    for char in [args.char]:
                        result = pool.apply_async(compare_pols, [policy, num_func, char, mem, args])
                        results.append(result)
        [result.wait() for result in results]
        for result in results:
            r = result.get()
            if r is not None:
                print(r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FaasCache Simulation')
    parser.add_argument("--tracedir", type=str, default="/data2/alfuerst/azure/functions/trace_pckl/represent/", required=False)
    parser.add_argument("--numfuncs", type=int, default=392, required=False)
    parser.add_argument("--savedir", type=str, default="/data2/alfuerst/verify-test/", required=False)
    parser.add_argument("--logdir", type=str, default="/data2/alfuerst/verify-test/logs/", required=False)
    parser.add_argument("--memstep", type=int, default=3000, required=False)
    # Defaults below reproduce the original hardcoded behaviour
    parser.add_argument("--memmin", type=int, default=1000, help="smallest memory size (MB)")
    parser.add_argument("--memmax", type=int, default=80000, help="memory sweep upper bound (MB, exclusive)")
    parser.add_argument("--char", type=str, default="b", help="trace variant label, i.e. the X in {numfuncs}-X.pckl")
    parser.add_argument("--workers", type=int, default=None, help="parallel simulations (default: all cores)")
    parser.add_argument("--delete-logs", action="store_true",
                        help="delete each run's per-invocation CSV logs once its result is saved "
                             "(they are only used by plotting/compute_mem_usage.py)")
    parser.add_argument("--policies", nargs="+", default=["GD", "TTL", "LRU", "LND", "HIST", "FREQ", "SIZE"],
                        help="policies to run; extension variants e.g. GDW_a1_b1_c0.5 GDD_t30 (see LambdaScheduler.parse_gd_variant)")
    args = parser.parse_args()
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    print(args)
    run_multiple_expts(args)
