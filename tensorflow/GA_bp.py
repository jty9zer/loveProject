from sko.GA import GA
import bp

ga = GA(func=bp.aim_function, n_dim=2, size_pop=16, max_iter=20, prob_mut=0.001, lb=[1, 0.001], ub=[20, 1], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
