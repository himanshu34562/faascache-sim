import os
import sys
import pandas as pd
from multiprocessing import Pool

# Make LambdaData importable no matter where this script is run from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LambdaData import *
import pickle
from math import ceil

# Paths are relative to the faascache-project repo root (sim-upstream is a submodule of it).
# Override with FAASCACHE_ROOT if your layout differs.
ROOT = os.environ.get("FAASCACHE_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
store = os.path.join(ROOT, "experiments/traces/split_funcs/")
buckets = [str(i) for i in range(1, 1441)]

datapath = os.path.join(ROOT, "azure-raw-data/")
durations = "function_durations_percentiles.anon.d01.csv"
invocations = "invocations_per_function_md.anon.d01.csv"
mem_fnames = "app_memory_percentiles.anon.d01.csv"
# Optional cap (0 = no cap). Set e.g. SPLIT_MAX_INVOCATIONS=10000000 on low-RAM machines.
MAX_INVOCS = int(os.environ.get("SPLIT_MAX_INVOCATIONS", "0"))

def trace_row(data):
    index, row = data
    out_pth = os.path.join(store, index+".pckl")
    if os.path.exists(out_pth):
        return  # resumable: already written by an earlier (possibly interrupted) run
    total = int(row[buckets].sum())
    if MAX_INVOCS and total > MAX_INVOCS:
        # One tuple per invocation: a 100M-call/day function needs >10GB RAM and kills the worker
        print("skipping {} ({} invocations/day > SPLIT_MAX_INVOCATIONS={})".format(index, total, MAX_INVOCS), flush=True)
        return
    secs_p_min = 60
    milis_p_sec = 1000
    trace = list()
    lambdas = dict()
    cold_dur = int(row["Maximum"])
    warm_dur = int(row["Average"])
    mem = int(row["divvied"])
    k = index
    d = LambdaData(k, mem, cold_dur, warm_dur)
    lambdas[k] = (k, mem, cold_dur, warm_dur)
    for minute, invocs in enumerate(row[buckets]):
        start = minute * secs_p_min * milis_p_sec
        if invocs == 0:
            continue
        elif invocs == 1:
            trace.append((d, start))
        else:
            every = (secs_p_min*milis_p_sec) / invocs
            trace += [(d, start + i*every) for i in range(invocs)]
    # print(trace)
    out_trace = sorted(trace, key=lambda x:x[1]) #(lamdata, t)

    # files are traces themselves. Just need to combine them together
    # format: (lambdas, trace) =>
    # lambdas:    dict[func_name] = (mem_size, cold_time, warm_time)
    # trace = [LambdaData(func_name, mem, cold_time, warm_time), start_time]
    # Write to a temp file then rename, so a killed worker never leaves a truncated .pckl behind
    tmp_pth = out_pth + ".tmp"
    with open(tmp_pth, "w+b") as f:
        pickle.dump((lambdas, out_trace), f)
    os.replace(tmp_pth, out_pth)
    # return lambdas, out_trace

def divive_by_func_num(row):
    return ceil(row["AverageAllocatedMb"] / group_by_app[row["HashApp"]])

file = os.path.join(datapath, durations.format(1))
durations = pd.read_csv(file)
durations.index = durations["HashFunction"]
durations = durations.drop_duplicates("HashFunction")

group_by_app = durations.groupby("HashApp").size()

file = os.path.join(datapath, invocations.format(1))
invocations = pd.read_csv(file)
invocations = invocations.dropna()
invocations.index = invocations["HashFunction"]
sums = invocations[buckets].sum(axis=1)
invocations = invocations[sums > 1] # action must be invoked at least twice
invocations = invocations.drop_duplicates("HashFunction")

joined = invocations.join(durations, how="inner", lsuffix='', rsuffix='_durs')

file = os.path.join(datapath, mem_fnames.format(1))
memory = pd.read_csv(file)
memory = memory.drop_duplicates("HashApp")
memory.index = memory["HashApp"]

new_mem = memory.apply(divive_by_func_num, axis=1, raw=False, result_type='expand')
memory["divvied"] = new_mem

joined = joined.join(memory, how="inner", on="HashApp", lsuffix='', rsuffix='_mems')

os.makedirs(store, exist_ok=True)
# 1 worker by default for low-memory machines; raise via SPLIT_WORKERS=4 etc.
with Pool(processes=int(os.environ.get("SPLIT_WORKERS", "1"))) as p:
    # small chunks: if a worker dies, only a few functions are lost (and a rerun resumes them)
    for _ in p.imap_unordered(trace_row, joined.iterrows(), chunksize=16):
        pass
