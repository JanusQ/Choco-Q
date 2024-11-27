should_print = True

from chocoq.problems.facility_location_problem import generate_flp
from chocoq.problems.set_cover_problem import generate_scp
from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.solvers.optimizers import CobylaOptimizer, AdamOptimizer
from chocoq.solvers.qiskit import (
    ChocoSolver, CyclicSolver, HeaSolver, PenaltySolver, ChocoSolverSearch, 
    AerGpuProvider, AerProvider, FakeBrisbaneProvider, FakeKyivProvider, FakeTorinoProvider, DdsimProvider,
)

num_case = 1
a, b = generate_scp(num_case,[(5, 5)])
# print(a[0][0])
# (1, [(2, 1), (3, 2), (3, 3), (4, 3), (4, 4)], 1, 20)

print(b)

best_lst = []
arg_lst = []

for i in range(num_case):
    opt = CobylaOptimizer(max_iter=200)
    aer = DdsimProvider()
    a[0][i].set_penalty_lambda(400)
    solver = ChocoSolver(
        prb_model=a[0][i],  # 问题模型
        optimizer=opt,  # 优化器
        provider=aer,  # 提供器（backend + 配对 pass_mannager ）
        num_layers=5,
        shots=1024,
        # mcx_mode="linear",
    )

    num, _, depth = solver.search()
    print(num, depth)
