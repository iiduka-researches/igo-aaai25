# Explicit and Implicit Graduated Optimization in Deep Neural Networks
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for image classification.  
nshb.py based on the implementation of "Understanding the Role of Momentum in Stochastic Gradient Methods" (NeurIPS2019).   
See <<https://github.com/Kipok/understanding-momentum>>.

# Abstract
Graduated optimization is a global optimization technique that is used to minimize a multimodal nonconvex function by smoothing the objective function with noise and gradually refining the solution. This paper experimentally evaluates the performance of the explicit graduated optimization algorithm with an optimal noise scheduling derived from a previous study with traditional benchmark functions and modern neural network architecture and discusses its limitations. In addition, this paper extends the implicit graduated optimization algorithm, which is based on the fact that stochastic noise in the optimization process of SGD implicitly smooths the objective function, to SGD with momentum, analyzes its convergence, and demonstrates its effectiveness through experiments on image classification tasks with ResNet architectures. The code is available at ¥url{https://anonymous.4open.science/r/go-25}.

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Explicit Graduated Optimization
One can run ego.py.
```
python3 ego.py
```

# Implicit Graduated Optimization
Please select method.
```
parser.add_argument('--method', default="constant", type=str, help="constant, lr, beta, lr-batch, beta-batch, lr-beta, cosine, exp")
```

"constant" means constant learning rate, batch size, and momentum.  
"lr" means only learning rate decayed, with constant batch size and momentum.  
"beta" means only momnetum decayed, with constant learning rate and batch size.  
"lr-batch" means lr decayed and batch size increased, with constant momentum.  
"beta-batch" means momentum decayed and batch size increased, with constant learning rate.  
"lr-beta" means lr decayed and momentum decayed, with constant batch size.  
"cosine" means cosine annealing learning rate schedule, with constant batch size and momentum.  
"exp" means exponential decay learning rate schedule, with constant batch size and momentum.

One can run main.py.
```
python3 main.py
```
