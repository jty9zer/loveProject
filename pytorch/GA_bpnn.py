import torch
from matplotlib import pyplot as plt
from sko.GA import GA
import bpnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_best_loss(size, iter, mut):
    ga = GA(func=bpnn.aim_function, n_dim=3, size_pop=size, max_iter=iter, prob_mut=mut, lb=[1, 0.001, 0.05],
            ub=[30, 0.1, 0.15],
            precision=[1, 0.001, 0.01])
    ga.to(device=device)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    return (best_y)


if __name__ == "__main__":
    train_times = 10  # 训练次数
    X = []
    Y = []
    for i in range(0, train_times):
        print("第%d次训练：" % i)
        X.append(i)
        Y.append(get_best_loss(16, 10, 0.01))
    plt.plot(X, Y, 'r')
    plt.show()
