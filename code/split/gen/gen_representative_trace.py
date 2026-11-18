import os
import sys
import argparse
import pandas as pd
from multiprocessing import Pool
import pickle
from math import ceil

# Make LambdaData importable no matter where this script is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LambdaData import *

# Paths are relative to the faascache-project repo root (sim-upstream is a submodule of it).
# Override with FAASCACHE_ROOT if your layout differs.
ROOT = os.environ.get("FAASCACHE_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
save_dir = os.path.join(ROOT, "experiments/traces/representative/")
two_hour_dir = os.path.join(ROOT, "experiments/traces/representative_two_hour/")
store = os.path.join(ROOT, "experiments/traces/split_funcs/")  # output of trace_split_funcs.py
buckets = [str(i) for i in range(1, 1441)]

datapath = os.path.join(ROOT, "azure-raw-data/")
durations = "function_durations_percentiles.anon.d01.csv"
invocations = "invocations_per_function_md.anon.d01.csv"
mem_fnames = "app_memory_percentiles.anon.d01.csv"

quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
def gen_trace(df, num_funcs, run):
    per_qant = num_funcs // 4
    sums = df[buckets].sum(axis=1)  # only the per-minute columns (pandas 2+ won't sum the string cols)
    qts = sums.quantile(quantiles)
    lambdas = {}
    trace = []
    save_pth = "{}-{}.pckl".format(num_funcs, run)
    save_pth = os.path.join(save_dir, save_pth)
    if not os.path.exists(save_pth):
        for i in range(4):
            low = qts.iloc[i]
            high = qts.iloc[i+1]
            choose_from = df[sums.between(low, high)]
            # print(choose_from.describe())
            # Seeded so the same (num_funcs, run) always gives the same trace on every machine
            seed = num_funcs * 1000 + ord(run) * 10 + i
            chosen = choose_from.sample(per_qant, random_state=seed)

            for index, row in chosen.iterrows():
                # print(index)
                pckl = "{}.pckl".format(index)
                path = os.path.join(store, pckl)
                with open(path, "r+b") as f:
                    data = pickle.load(f)
                    one_lambda, one_trace = data
                    lambdas = {**lambdas, **one_lambda}
                    trace += one_trace

        out_trace = sorted(trace, key=lambda x:x[1]) #(lamdata, t)
        print(num_funcs, len(out_trace))
        with open(save_pth, "w+b") as f:
            # format: (lambdas, trace) => 
            # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
            # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
            data = (lambdas, out_trace)
            pickle.dump(data, f)

    print("done", save_pth)

def convert_to_two_hour(path):
    noon_ms = 12 * 60 * 60 * 1000
    two_pm_ms = 14 * 60 * 60 * 1000

    num_funcs, run = path.split("/")[-1][:-5].split("-")
    save_pth = "{}-{}.pckl".format(num_funcs, run)
    save_pth = os.path.join(two_hour_dir, save_pth)

    if not os.path.exists(save_pth):
        with open(path, "r+b") as f:
            # format: (lambdas, trace) => 
            # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
            # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
            lambdas, trace = pickle.load(f)

        start = stop = 0
        for i, (d, t) in enumerate(trace):
            if t >= noon_ms:
                start = i
                break

        for i, (d, t) in enumerate(trace[start:]):
            if t > two_pm_ms:
                stop = i + start
                break

        save_trc = trace[start:stop]
        # print(start, stop, len(trace), len(save_trc))
        with open(save_pth, "w+b") as f:
            # format: (lambdas, trace) => 
            # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
            # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
            data = (lambdas, save_trc)
            pickle.dump(data, f)
        
    print("done", save_pth)

def gen_traces(sizes, runs, max_func_mb=0):
    global durations
    global invocations
    global memory

    def divive_by_func_num(row):
        return ceil(row["AverageAllocatedMb"] / group_by_app[row["HashApp"]])

    file = os.path.join(datapath, durations)
    durations = pd.read_csv(file)
    durations.index = durations["HashFunction"]
    durations = durations.drop_duplicates("HashFunction")

    group_by_app = durations.groupby("HashApp").size()

    file = os.path.join(datapath, invocations)
    invocations = pd.read_csv(file)
    invocations = invocations.dropna()
    invocations.index = invocations["HashFunction"]
    sums = invocations[buckets].sum(axis=1)

    invocations = invocations[sums > 1] # action must be invoked at least twice
    invocations = invocations.drop_duplicates("HashFunction")
    # sums = invocations.sum(axis=1)
    # print(sums.quantile(quantiles))

    joined = invocations.join(durations, how="inner", lsuffix='', rsuffix='_durs')

    file = os.path.join(datapath, mem_fnames.format(1))
    memory = pd.read_csv(file)
    memory = memory.drop_duplicates("HashApp")
    memory.index = memory["HashApp"]

    new_mem = memory.apply(divive_by_func_num, axis=1, raw=False, result_type='expand')
    memory["divvied"] = new_mem

    joined = joined.join(memory, how="inner", on="HashApp", lsuffix='', rsuffix='_mems')

    # Only sample functions that stage 1 actually produced a split pickle for
    have = pd.Series([os.path.exists(os.path.join(store, "{}.pckl".format(h))) for h in joined.index],
                     index=joined.index)
    missing = int((~have).sum())
    if missing:
        print("warning: {} of {} functions have no split pickle in {}, skipping them".format(
            missing, len(joined), store))
    joined = joined[have]

    if max_func_mb:
        # Running containers can't be evicted, so a function whose *own* concurrent demand exceeds
        # what a server can hold swamps memory and makes every keep-alive policy look identical.
        # Little's law at the function's busiest minute: concurrency ~= rate/s * duration (s).
        # Uses the cold-start time (Maximum, which is what the simulator runs a cold start for):
        # while one cold start runs, every new arrival also goes cold, so this is the burst the
        # function needs after any eviction. Using Average alone still left 50%+ requests dropped.
        peak_rate = joined[buckets].max(axis=1) / 60.0
        demand_mb = peak_rate * (joined["Maximum"] / 1000.0) * joined["divvied"]
        too_big = demand_mb > max_func_mb
        print("excluding {} of {} functions whose peak cold-start burst demand > {} MB "
              "(largest: {:.0f} MB)".format(int(too_big.sum()), len(joined), max_func_mb, demand_mb.max()))
        joined = joined[~too_big]

    os.makedirs(save_dir, exist_ok=True)
    for size in sizes:
        for run in runs:
            gen_trace(joined, size, run)

if __name__ == "__main__":
    # Upstream generated every size in range(8, 400, 4) x runs a/b/c (~290 traces).
    # Defaults to just the paper's 392-function trace; pass --sizes/--runs for more.
    parser = argparse.ArgumentParser(description="Build representative traces from split_funcs pickles")
    parser.add_argument("--sizes", type=int, nargs="+", default=[392],
                        help="number of functions per trace (multiples of 4)")
    parser.add_argument("--runs", nargs="+", default=["a"],
                        help="single-letter variant labels, e.g. a b c")
    parser.add_argument("--max-func-mb", type=int, default=2048,
                        help="exclude functions whose peak cold-start burst memory (peak rate x Maximum duration x mem) "
                             "exceeds this many MB; 0 disables the filter")
    parser.add_argument("--two-hour", action="store_true",
                        help="also write 12:00-14:00 cuts of each trace to representative_two_hour/")
    args = parser.parse_args()

    gen_traces(args.sizes, args.runs, args.max_func_mb)

    if args.two_hour:
        os.makedirs(two_hour_dir, exist_ok=True)
        for size in args.sizes:
            for run in args.runs:
                convert_to_two_hour(os.path.join(save_dir, "{}-{}.pckl".format(size, run)))
