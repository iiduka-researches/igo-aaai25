import numpy as np
import time
import functions
import os

def gradient(f, x, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(x.shape[1]):
        x0 = x.copy()
        x1 = x.copy()
        x0[:, i] -= epsilon
        x1[:, i] += epsilon
        grad[:, i] = (f(x1) - f(x0)) / (2 * epsilon)
    return grad

def gradient_descent(f, delta, x_init, learning_rate=0.01, num_steps=40):
    x = x_init.copy()
    path = [x.copy()]
    num_samples = 100
    
    for step in range(num_steps):
        grad = np.zeros_like(x)
        for i in range(num_samples):
            noise = delta * np.random.randn(*x.shape)
            x_noisy = x + noise
            grad += gradient(f, x_noisy)
        grad /= num_samples
        x -= learning_rate * grad
        path.append(x.copy())
    
    return x

def geometric_sequence(a, b, c):
    r = (b / a) ** (1 / (c - 1))
    return a * r ** np.arange(c)

delta_scale = "nice"
M=50
delta_init = 1.0
delta_end = 0.01
if delta_scale == "geo":
    deltas = geometric_sequence(delta_init, delta_end, M+1)
elif delta_scale == "nice":
    deltas = []
    delta = delta_init
    for m in range(M):
        deltas.append(delta)
        gamma = (M-m)/(M-(m-1))
        delta *= gamma
    deltas.append(delta_end)

D=50
if D==50:
    func_names = ["ackley_func", "alpine1_func", "drop_wave_func", "ellipsoid_func", "hgbat_func", "modridge_func", "rastrigin_func",
                 "rosenbrock_func", "rothellipsoid_func", "schafferf7_func", "schwefel_func", "sphere_func", "griewank_func", "happycat_func", "salomon_func", "schwefel221_func"]
    l_0_values = {"ackley_func": 10, "alpine1_func": 1, "drop_wave_func": 0.1, "ellipsoid_func": 0.01, "hgbat_func": 0.1, "modridge_func": 1,
                "rastrigin_func": 0.01, "rosenbrock_func": 0.00005, "rothellipsoid_func": 0.01, "schafferf7_func": 20, "schwefel_func": 10,
                "sphere_func": 1, "griewank_func":50, "happycat_func":10, "salomon_func":10, "schwefel221_func":10}

func_names = ["rastrigin_func"]
os.makedirs('./results', exist_ok=True)
for func_name in func_names:
    original_f = functions.get_function(func_name)
    l_0 = l_0_values[func_name]

    algorithm = "ego"
    opt_list = []
    log_lines = []
    LB, UB = functions.get_initial_point(func_name, 0)
    start_time = time.time()
    for rand in range(1):
        print(rand)
        log_lines.append(f"{rand}")
        np.random.seed(rand)
        x_init = functions.get_initial_point(func_name, D)

        if algorithm == "ego":
            x = x_init.copy()
            for i in range(M+1):
                x = gradient_descent(original_f, deltas[i], x, learning_rate=(l_0)*deltas[i], num_steps=20)
                log_line = f"delta[{i}]: {deltas[i]}, Optimal value: {original_f(x)}"
                log_lines.append(log_line)
            m, s = divmod(int(time.time()-start_time), 60)
            print(f"{m}m{s}s")
            opt_list.append(original_f(x))  

    log_lines.append(f"ave:{np.mean(opt_list)}, \nmean:{np.median(opt_list)}, \nstd:{np.std(opt_list)}")
    m, s = divmod(int(time.time()-start_time), 60)
    log_lines.append(f"{m}m{s}s")
    log_lines.append(f"(lr={l_0}*deltas[i]")

    txt_name = f"./results/{func_name}.txt"
    with open(txt_name, "w") as log_file:
        for line in log_lines:
            log_file.write(line + "\n")