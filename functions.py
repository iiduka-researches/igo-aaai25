import numpy as np

def ellipsoid_func(x):
    D = x.shape[1]
    Array1 = np.arange(1, D + 1)
    return np.sum(x**2 * Array1, axis=1)

def ackley_func(x):
    D = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * x), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / D)) - np.exp(sum2 / D) + np.exp(1) + 20

def alpine1_func(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

def bent_cigar_func(x):
    D = x.shape[1]
    x2toend = x[:, 1:D]
    sum1 = np.sum(x2toend**2, axis=1)
    return x[:, 0]**2 + 1000000 * sum1

def discus_func(x):
    D = x.shape[1]
    x2toend = x[:, 1:D]
    sum1 = np.sum(x2toend**2, axis=1)
    return 1000000 * x[:, 0]**2 + sum1

def drop_wave_func(x):
    sum1 = np.sum(x**2, axis=1)
    return 1 - (1 + np.cos(12 * np.sqrt(sum1))) / (0.5 * sum1 + 2)

def ellipt_func(x):
    D = x.shape[1]
    array1 = np.arange(1, D + 1)
    sum1 = np.sum((1000000)**((array1 - 1) / (D - 1)) * x**2, axis=1)
    return sum1

def expschafferf6_func(x):
    D = x.shape[1]
    g = lambda z, y: 0.5 + (np.sin(np.sqrt(z**2 + y**2))**2 - 0.5) / (1 + 0.001 * (z**2 + y**2))**2
    sum1 = 0
    for i in range(D - 1):
        sum1 += g(x[:, i], x[:, i + 1])
    return sum1 + g(x[:, D - 1], x[:, 0])

def griewank_func(x):
    D = x.shape[1]
    array1 = np.arange(1, D + 1)
    sum1 = np.sum(x**2, axis=1) / 4000
    prod1 = np.prod(np.cos(x / np.sqrt(array1)), axis=1)
    return sum1 - prod1 + 1

def happycat_func(x):
    D = x.shape[1]
    sum1 = np.sum(x, axis=1)
    sum2 = np.sum(x**2, axis=1)
    return (np.abs(sum2 - D))**0.25 + (0.5 * sum2 + sum1) / D + 0.5

def hgbat_func(x):
    D = x.shape[1]
    sum1 = np.sum(x, axis=1)
    sum2 = np.sum(x**2, axis=1)
    return np.sqrt(np.abs(sum2**2 - sum1**2)) + (0.5 * sum2 + sum1) / D + 0.5

def modridge_func(x):
    x2 = x**2
    x2 = np.delete(x2, 0, axis=1)
    sum1 = np.sum(x2, axis=1)
    return np.abs(x[:, 0]) + 2 * sum1**0.1

def modxinsyang3_func(x):
    sum1 = np.sum((x / 15)**10, axis=1)
    sum2 = np.sum(x**2, axis=1)
    return 1e4 * (1 + (np.exp(-sum1) - 2 * np.exp(-sum2)) * np.prod(np.cos(x)**2, axis=1))

def modxinsyang5_func(x):
    sum1 = np.sum(np.sin(x)**2, axis=1)
    sum2 = np.sum(x**2, axis=1)
    sum3 = np.sum(np.sin(np.sqrt(np.abs(x)))**2, axis=1)
    return 1e4 * (1 + (sum1 - np.exp(-sum2)) * np.exp(-sum3))

def permdb_func(x):
    D = x.shape[1]
    Array1 = np.arange(1, D + 1)
    sum2 = 0
    for i in range(1, D + 1):
        sum1 = np.sum((Array1**i + 0.5) * ((x / Array1)**i - 1), axis=1)
        sum2 += sum1**2
    return sum2

def quintic_func(x):
    return np.sum(np.abs(x**5 - 3*x**4 + 4*x**3 + 2*x**2 - 10*x - 4), axis=1)

def rastrigin_func(x):
    D = x.shape[1]
    sum1 = np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)
    return sum1 + 10 * D

def rosenbrock_func(x):
    D = x.shape[1]
    xi = x[:, :D - 1]
    xiplus1 = x[:, 1:D]
    sum1 = np.sum(100 * (xiplus1 - xi**2)**2 + (xi - 1)**2, axis=1)
    return sum1

def rothellipsoid_func(x):
    D = x.shape[1]
    Array1 = np.arange(1, D + 1)
    sum1 = np.sum((D + 1 - Array1) * x**2, axis=1)
    return sum1

def salomon_func(x):
    sum1 = np.sum(x**2, axis=1)
    return 1 - np.cos(2 * np.pi * np.sqrt(sum1)) + 0.1 * np.sqrt(sum1)

def schafferf7_func(x):
    D = x.shape[1]
    xi = x[:, :-1]
    xiplus1 = x[:, 1:]
    si = np.sqrt(xi**2 + xiplus1**2)
    sum1 = np.sum(np.sqrt(si) + np.sqrt(si) * (np.sin(50 * si**0.2))**2, axis=1)
    return (1 / (D - 1) * sum1)**2

def schwefel221_func(x):
    return np.max(np.abs(x), axis=1)

def schwefel222_func(x):
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def schwefel_func(x):
    D = x.shape[1]
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1) + 418.9828872724337 * D

def sphere_func(x):
    return np.sum(x**2, axis=1)

def sumpow2_func(x):
    D = x.shape[1]
    Array1 = np.arange(1, D + 1)
    return np.sum(np.abs(x)**(2 + 4 * (Array1 - 1) / (D - 1)), axis=1)

def sumpow_func(x):
    D = x.shape[1]
    Array1 = np.arange(2, D + 2)
    return np.sum(np.abs(x)**Array1, axis=1)

def weierstrass_func(x):
    D = x.shape[1]
    a = 0.5
    b = 3
    k_max = 20

    kVec = np.arange(0, k_max + 1)
    sum2 = np.sum(a**kVec * np.cos(np.pi * b**kVec))

    f = 0
    for i in range(D):
        sum1 = np.sum(a**kVec * np.cos(2 * np.pi * b**kVec * (x[:, i] + 0.5)))
        f += sum1

    y = f - D * sum2
    return y


def xinsheyang1_func(x):
    sum1 = np.sum(np.abs(x), axis=1)
    sum2 = np.sum(np.sin(x**2), axis=1)
    return sum1 * np.exp(-sum2)

def zakharov_func(x):
    D = x.shape[1]
    Array1 = np.arange(1, D + 1)
    sum1 = np.sum(x**2, axis=1)
    sum2 = np.sum(0.5 * Array1 * x, axis=1)
    return sum1 + sum2**2 + sum2**4

def levy_func(x):
    D = x.shape[1]
    z = 1 + (x - 1) / 4
    s = np.sin(np.pi * z[:, 0]) ** 2
    
    for i in range(D - 1):
        s += (z[:, i] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * z[:, i] + 1)) ** 2)
    
    s += (z[:, D - 1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * z[:, D - 1])) ** 2)
    return s

def michalewicz_func(x):
    D = x.shape[1]
    m = 10
    s = 0
    for i in range(1, D + 1):
        s += np.sin(x[:, i - 1]) * (np.sin(i * x[:, i - 1] ** 2 / np.pi)) ** (2 * m)
    return -s

def get_function(func_name):
    functions = {
        "ackley_func": ackley_func,
        "alpine1_func": alpine1_func,
        "bent_cigar_func": bent_cigar_func,
        "discus_func": discus_func,
        "drop_wave_func": drop_wave_func,
        "ellipsoid_func": ellipsoid_func,
        "ellipt_func": ellipt_func,
        "expschafferf6_func": expschafferf6_func,
        "griewank_func": griewank_func,
        "happycat_func": happycat_func,
        "hgbat_func": hgbat_func,
        "modridge_func": modridge_func,
        "modxinsyang3_func": modxinsyang3_func,
        "modxinsyang5_func": modxinsyang5_func,
        "permdb_func": permdb_func,
        "quintic_func": quintic_func,
        "rastrigin_func": rastrigin_func,
        "rosenbrock_func": rosenbrock_func,
        "rothellipsoid_func": rothellipsoid_func,
        "salomon_func": salomon_func,
        "schafferf7_func": schafferf7_func,
        "schwefel221_func": schwefel221_func,
        "schwefel222_func": schwefel222_func,
        "schwefel_func": schwefel_func,
        "sphere_func": sphere_func,
        "sumpow2_func": sumpow2_func,
        "sumpow_func": sumpow_func,
        "weierstrass_func": weierstrass_func,
        "xinsheyang1_func": xinsheyang1_func,
        "zakharov_func": zakharov_func,
        "levy_func": levy_func,
        "michalewicz_func": michalewicz_func
    }
    
    return functions.get(func_name, None)

def get_initial_point(func_name, D):
    search_ranges = {
        "ackley_func": (-32.768, 32.768),
        "alpine1_func": (-10, 10),
        "bent_cigar_func": (-100, 100),
        "discus_func": (-100, 100),
        "drop_wave_func": (-5.12, 5.12),
        "ellipsoid_func": (-100, 100),
        "ellipt_func": (-100, 100),
        "expschafferf6_func": (-100, 100),
        "griewank_func": (-100, 100),
        "happycat_func": (-20, 20),
        "hgbat_func": (-15, 15),
        "modridge_func": (-100, 100),
        "modxinsyang3_func": (-20, 20),
        "modxinsyang5_func": (-100, 100),
        "permdb_func": (-50, 50),
        "quintic_func": (-20, 20),
        "rastrigin_func": (-5.12, 5.12),
        "rosenbrock_func": (-10, 10),
        "rothellipsoid_func": (-100, 100),
        "salomon_func": (-20, 20),
        "schafferf7_func": (-100, 100),
        "schwefel221_func": (-100, 100),
        "schwefel222_func": (-100, 100),
        "schwefel_func": (-500, 500),
        "sphere_func": (-100, 100),
        "sumpow2_func": (-10, 10),
        "sumpow_func": (-10, 10),
        "weierstrass_func": (-0.5, 0.5),
        "xinsheyang1_func": (-2 * np.pi, 2 * np.pi),
        "zakharov_func": (-10, 10),
        "levy_func": (-10, 10),
        "michalewicz_func": (0, np.pi)
    }

    if D == 0:
        return search_ranges[func_name]

    if func_name in search_ranges:
        lower_bound, upper_bound = search_ranges[func_name]
        return np.random.uniform(lower_bound, upper_bound, (1, D))
    else:
        raise ValueError(f"The search range for function '{func_name}' is not defined.")