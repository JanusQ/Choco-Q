# nohup python3 2_table_all_scale.py > 2_table_all_scale.output 2>&1 &
import os
import time
import csv
import signal
import random
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from chocoq.problems.facility_location_problem import generate_flp
from chocoq.problems.graph_coloring_problem import generate_gcp
from chocoq.problems.k_partition_problem import generate_kpp
from chocoq.problems.job_scheduling_problem import generate_jsp
from chocoq.problems.traveling_salesman_problem import generate_tsp
from chocoq.problems.set_cover_problem import generate_scp
from chocoq.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from chocoq.solvers.qiskit import (
    PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver, 
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

import pandas as pd
pd.set_option('display.max_rows', None)  # display all rows
pd.set_option('display.max_columns', None)  # display all columns.

num_cases = 10  # The number of cases in each benchmark
problem_scale = 4 # The problem scale, 1 is the minimal scale like F1,K1,G1 in Table 1 of paper
#2 means F2 K2 ... In CPU version, this benchmarks with higher scale is much slower when solving with baselines.

flp_problems_pkg, flp_configs_pkg = generate_flp(num_cases, [(1, 2), (2, 3), (3, 3), (3, 4)][:problem_scale], 1, 20)
gcp_problems_pkg, gcp_configs_pkg = generate_gcp(num_cases, [(3, 1), (3, 2), (4, 1), (4, 2)][:problem_scale])
kpp_problems_pkg, kpp_configs_pkg = generate_kpp(num_cases, [(4, 2, 3), (5, 3, 4), (6, 3, 5), (7, 3, 6)][:problem_scale], 1, 20)
jsp_problems_pkg, jsp_configs_pkg = generate_jsp(num_cases, [(2, 2, 3), (2, 3, 4), (3, 3, 5), (3, 4, 6)][:problem_scale], 1, 20)
scp_problems_pkg, scp_configs_pkg = generate_scp(num_cases, [(4, 4), (5, 5), (6, 6), (7, 7)][:problem_scale])

configs_pkg = flp_configs_pkg + gcp_configs_pkg + kpp_configs_pkg + jsp_configs_pkg + scp_configs_pkg
with open(f"2_table_all_scale.config", "w") as file:
    for pkid, configs in enumerate(configs_pkg):
        for problem in configs:
            file.write(f'{pkid}: {problem}\n')

new_path = '2_table_depth_all_scale'

problems_pkg = flp_problems_pkg + gcp_problems_pkg + kpp_problems_pkg + jsp_problems_pkg + scp_problems_pkg

metrics_lst = ['culled_depth', 'num_params']
solvers = [PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver]
headers = ["pkid", 'method', 'layers'] + metrics_lst

def process_layer(prb, num_layers, solver, metrics_lst):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        # Select CPU or GPU simulator
        # cpu simulator, comment it when use GPU
        # provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        # uncomment the line below to use GPU simulator
        provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    metrics = used_solver.circuit_analyze(metrics_lst)
    return metrics

if __name__ == '__main__':
    set_timeout = 60 * 60 * 24 # Set timeout duration
    num_complete = 0
    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers once

    num_processes_cpu = os.cpu_count() // 2
    with ProcessPoolExecutor(max_workers=num_processes_cpu) as executor:
        futures = []
        for solver in solvers:
            for pkid, problems in enumerate(problems_pkg):
                for problem in problems:
                    if solver == ChocoSolver:
                        num_layers = 1
                    else:
                        num_layers = 5
                    future = executor.submit(process_layer, problem, num_layers, solver, metrics_lst)
                    futures.append((future, pkid, solver.__name__, num_layers))

        start_time = time.perf_counter()
        for future, pkid, solver, num_layers in futures:
            current_time = time.perf_counter()
            remaining_time = max(set_timeout - (current_time - start_time), 0)
            diff = []
            try:
                result = future.result(timeout=remaining_time)
                diff.extend(result)
                print(f"Task for problem {pkid}, num_layers {num_layers} executed successfully.")
            except MemoryError:
                diff.append('memory_error')
                print(f"Task for problem {pkid}, num_layers {num_layers} encountered a MemoryError.")
            except TimeoutError:
                diff.append('timeout')
                print(f"Task for problem {pkid}, num_layers {num_layers} timed out.")
            finally:
                row = [pkid, solver, num_layers] + diff
                with open(f'{new_path}.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)  # Write row immediately
                num_complete += 1
                if num_complete == len(futures):
                    print(f'Data has been written to {new_path}.csv')
                    for process in executor._processes.values():
                        os.kill(process.pid, signal.SIGTERM)

file_path = '2_table_depth_all_scale.csv'

df = pd.read_csv(file_path)

grouped_df = df.groupby(['pkid', 'layers', 'method'], as_index=False).agg({
    "culled_depth": 'mean',
})

values = ["culled_depth"]
pivot_df = grouped_df.pivot(index =['pkid'], columns='method', values=values)

method_order = ['PenaltySolver', 'CyclicSolver', 'HeaSolver', 'ChocoSolver']
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([values, method_order]))

print(pivot_df)

new_path = '2_table_evaluate_all_scale'

problems_pkg = list(
    itertools.chain(
        enumerate(flp_problems_pkg),
        enumerate(gcp_problems_pkg),
        enumerate(kpp_problems_pkg),
        enumerate(jsp_problems_pkg),
        enumerate(scp_problems_pkg),
    )
)

solvers = [PenaltySolver, CyclicSolver, HeaSolver, ChocoSolver]
evaluation_metrics = ['best_solution_probs', 'in_constraints_probs', 'ARG', 'iteration_count', 'classcial', 'quantum', 'run_times']
headers = ['pkid', 'pbid', 'layers', "variables", 'constraints', 'method'] + evaluation_metrics

def process_layer(prb, num_layers, solver):
    opt = CobylaOptimizer(max_iter=200)
    ddsim = DdsimProvider()
    cpu = AerProvider()
    gpu = AerGpuProvider()
    prb.set_penalty_lambda(400)
    used_solver = solver(
        prb_model = prb,
        optimizer = opt,
        # 根据配置的环境选择CPU或GPU
        # provider = cpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        provider = gpu if solver in [PenaltySolver, CyclicSolver, HeaSolver] else ddsim,
        num_layers = num_layers,
        shots = 1024,
    )
    used_solver.solve()
    eval = used_solver.evaluation()
    time = list(used_solver.time_analyze())
    run_times = used_solver.run_counts()
    return eval + time + [run_times]

if __name__ == '__main__':
    all_start_time = time.perf_counter()
    set_timeout = 60 * 60 * 2 # Set timeout duration
    num_complete = 0

    with open(f'{new_path}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入标题

    num_processes_cpu = os.cpu_count()
    # pkid-pbid: 问题包序-包内序号
    for pkid, (diff_level, problems) in enumerate(problems_pkg):
        for solver in solvers:
            if solver in [PenaltySolver, CyclicSolver, HeaSolver]:
                num_processes = 2**(4 - diff_level) + 1
            else:
                num_processes = 100

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = []
                if solver == ChocoSolver:
                    layer = 1
                else:
                    layer = 5

                for pbid, prb in enumerate(problems):
                    print(f'{pkid}-{pbid}, {layer}, {solver} build')
                    future = executor.submit(process_layer, prb, layer, solver)
                    futures.append((future, prb, pkid, pbid, layer, solver.__name__))

                start_time = time.perf_counter()
                for future, prb, pkid, pbid, layer, solver in futures:
                    current_time = time.perf_counter()
                    remaining_time = max(set_timeout - (current_time - start_time), 0)
                    diff = []
                    try:
                        metrics = future.result(timeout=remaining_time)
                        diff.extend(metrics)
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} executed successfully.")
                    except MemoryError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} encountered a MemoryError.")
                        for dict_term in evaluation_metrics:
                            diff.append('memory_error')
                    except TimeoutError:
                        print(f"Task for problem {pkid}-{pbid} L={layer} {solver} timed out.")
                        for dict_term in evaluation_metrics:
                            diff.append('timeout')
                    except Exception as e:
                        print(f"An error occurred: {e}")
                    finally:
                        row = [pkid, pbid, layer, len(prb.variables), len(prb.lin_constr_mtx), solver] + diff
                        with open(f'{new_path}.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row)  # Write row immediately
                        num_complete += 1
                        if num_complete == len(futures):
                            print(f'problem_pkg_{pkid} has finished')
                            for process in executor._processes.values():
                                os.kill(process.pid, signal.SIGTERM)
    print(f'Data has been written to {new_path}.csv')
    print(time.perf_counter()- all_start_time)

df = pd.read_csv('2_table_evaluate_all_scale.csv')

df = df.drop(columns=['pbid'])
df = df[df['ARG'] <= 100000]

grouped_df = df.groupby(['pkid', 'layers', 'variables', 'constraints', 'method'], as_index=False).agg({
    "ARG": 'mean',
    'in_constraints_probs': 'mean',
    'best_solution_probs': 'mean',
    'iteration_count':'mean',
    'classcial':'mean',
    'run_times':'mean',
})

## 分组并把组作为索引
pivot_df = grouped_df.pivot(index =['pkid', 'variables', 'constraints'], columns='method', values=["best_solution_probs",'in_constraints_probs', 'ARG'])
method_order = ['PenaltySolver', 'CyclicSolver', 'HeaSolver', 'ChocoSolver']
pivot_df = pivot_df.reindex(columns=pd.MultiIndex.from_product([["best_solution_probs",'in_constraints_probs', 'ARG'], method_order]))

print(pivot_df)

# Row-wise improvement calculation
methods = ['CyclicSolver']
improvements_rowwise = []

for idx, row in pivot_df.iterrows():
    choco_best_solution_probs = row['best_solution_probs', 'ChocoSolver']
    choco_in_constraints_probs = row['in_constraints_probs', 'ChocoSolver']
    choco_ARG = row['ARG', 'ChocoSolver']

    for method in methods:
        # Avoid division by zero and infinite values
        if row['best_solution_probs', method] != 0 and row['in_constraints_probs', method] != 0 and choco_ARG != 0:
            improvement_best_solution_probs = choco_best_solution_probs / row['best_solution_probs', method]
            improvement_in_constraints_probs = choco_in_constraints_probs / row['in_constraints_probs', method]
            improvement_ARG = row['ARG', method] / choco_ARG

            # Check for finite values
            if np.isfinite(improvement_best_solution_probs) and np.isfinite(improvement_in_constraints_probs) and np.isfinite(improvement_ARG):
                improvements_rowwise.append({
                    'pkid': row.name[0], 
                    'variables': row.name[1], 
                    'constraints': row.name[2],
                    'method': method,
                    'improvement_best_solution_probs': improvement_best_solution_probs,
                    'improvement_in_constraints_probs': improvement_in_constraints_probs,
                    'improvement_ARG': improvement_ARG
                })

# Convert the result into a DataFrame
improvements_rowwise_df = pd.DataFrame(improvements_rowwise)

# Calculate the average improvement for each metric
improvements_avg_df = improvements_rowwise_df.groupby('method').mean()[['improvement_best_solution_probs', 'improvement_in_constraints_probs', 'improvement_ARG']]
print(improvements_avg_df)
