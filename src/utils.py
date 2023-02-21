import numpy as np
import time
from scipy.optimize import minimize
import matplotlib

def save_fig(fig, path):
    bc = matplotlib.get_backend()
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    fig.savefig(path, bbox_inches='tight')
    matplotlib.use(bc)


class Grad:
    def __init__(self, f, x, iter, gradient_method=None):
        self.f = f
        self.x = x
        self.n = len(x)
        self.iter = iter
        self.gradient_method = gradient_method

    def __call__(self, *args, **kwargs):
        grad = np.zeros_like(self.x)
        if self.gradient_method == 'parameter_shift':
            for index in np.arange(self.n):
                unit_vec = np.array([1.0 if i == index else 0.0 for i in range(self.n)])
                x_plus = self.x + (np.pi / 2) * unit_vec
                x_minus = self.x - (np.pi / 2) * unit_vec
                grad[index] = (self.f(x_plus) - self.f(x_minus)) / 2
        elif self.gradient_method == 'finite_differences':
            shift = np.pi / 2
            for index in np.arange(self.n):
                unit_vec = np.array([1.0 if i == index else 0.0 for i in range(self.n)])
                x_plus = self.x + shift * unit_vec
                x_minus = self.x - shift * unit_vec
                grad[index] = (self.f(x_plus) - self.f(x_minus)) / (2 * shift)
        elif self.gradient_method == 'spsa':
            const = np.pi / 5
            const = const / (self.iter + 1)**.5
            pertubation = np.random.choice([-1, 1], self.n, p=[.5, .5])
            x_plus = self.x + const * pertubation
            x_minus = self.x - const * pertubation
            grad = (self.f(x_plus) - self.f(x_minus)) / (2 * const * pertubation)
        else:
            assert False, "Gradient 'gradient_method' should be something from ['parameter_shift', 'finite_differences', 'spsa']"

        return grad


class Optimizer:
    def __init__(self, f, grad, f_ratio, num_variables, alpha_const=1.,
                 max_iter=10, gradient_method=None, optimization_step='vanilla',
                 backtracking=False, fine_tuning=False):
        self.iter = 0
        self.max_iter = max_iter
        self.alpha_const = alpha_const
        self.gradient_method = gradient_method
        self.optimization_step = optimization_step
        self.backtracking = backtracking
        self.fine_tuning = fine_tuning

        self.tol = 1e-10
        self.vanishing_grad_flag = False

        self.f = f
        self.grad = grad
        self.f_ratio = f_ratio
        self.num_variables = num_variables
        self.x = None
        self.history = {'loss': [], 'norm_grad': [], 'ratio': [], 'index': []}
        self.time = 0

        # For Adam
        self.mt = np.zeros((self.num_variables,))
        self.vt = np.zeros((self.num_variables,))
        self.beta1 = .9
        self.beta2 = .999
        self.eps = 1e-10

    def get_alpha(self, *args, **kwargs):
        alpha = self.armijo(*args, **kwargs) if self.backtracking else np.sqrt(self.num_variables/(self.iter+1))
        return alpha

    def armijo(self, val, grad, search_direction, x, rho=.5, c1=1e-4):
        """
        Armijo Backtracking Line search
        """
        alpha = 1
        descent = grad.T @ search_direction
        max_iter_armijo = 3
        for lsiter in range(max_iter_armijo):
            val_neu = self.f(x + alpha * search_direction)
            if val_neu <= val + c1 * alpha * descent:
                return alpha
            alpha = rho * alpha
        print(f"Line search exited on time after {max_iter_armijo} iterations.")
        return alpha

    def step_end(self, x, alpha, norm_grad):
        val = self.f(x)
        ratio, index = self.f_ratio(x)
        self.history['loss'].append(val)
        self.history['norm_grad'].append(norm_grad)
        self.history['ratio'].append(ratio)
        self.history['index'].append(index)
        print(f'Iter {self.iter: <4}, alpha = {np.round(alpha, 2): <6}, '
              f'norm_grad = {np.round(norm_grad, 2): <10}, cost = {np.round(val, 4): <10}')
        return

    def vanilla_step(self, x, gradient_method):
        val = self.f(x)
        grad = self.grad(x, self.iter, gradient_method=gradient_method)
        norm_grad = np.linalg.norm(grad)
        if norm_grad <= self.tol:
            self.vanishing_grad_flag = True
            print(f"Optimization ended on time due to vanishing gradient.")
            return x
        grad = grad / norm_grad
        search_direction = -grad
        alpha = self.get_alpha(val, grad, search_direction, x)
        x = x + alpha * search_direction
        self.step_end(x, alpha, norm_grad)
        return x

    def adam_step(self, x, gradient_method):
        val = self.f(x)
        grad = self.grad(x, self.iter, gradient_method=gradient_method)
        norm_grad = np.linalg.norm(grad)
        if norm_grad <= self.tol:
            self.vanishing_grad_flag = True
            print(f"Optimization ended on time due to vanishing gradient.")
            return x
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * grad**2
        mt_hat = self.mt / (1 - self.beta1**(self.iter + 1))
        vt_hat = self.vt / (1 - self.beta2**(self.iter + 1))
        search_direction = - mt_hat / (np.sqrt(vt_hat) + self.eps)
        alpha = self.get_alpha(val, grad, search_direction, x)
        x = x + alpha * search_direction
        self.step_end(x, alpha, norm_grad)
        return x

    def cobyla(self, x):
        res = minimize(self.f, x0=x, tol=self.tol, method='COBYLA')
        x = res.x
        ratio, index = self.f_ratio(x)
        val = self.f(x)
        self.history['loss'].append(val)
        self.history['norm_grad'].append(np.linalg.norm(self.grad(x, self.iter, gradient_method='parameter_shift')))
        self.history['ratio'].append(ratio)
        self.history['index'].append(index)
        print(res)
        return x

    def optimize(self, x):
        self.time = time.time()
        self.history['loss'].append(self.f(x))
        self.history['norm_grad'].append(np.linalg.norm(self.grad(x, self.iter, gradient_method='parameter_shift')))
        self.history['ratio'].append(self.f_ratio(x)[0])
        self.history['index'].append(self.f_ratio(x)[1])

        if self.gradient_method is None:
            print('cobyla optimization'.upper())
            x = self.cobyla(x)
        else:
            print(' '.join([x.upper() for x in self.gradient_method.split('_')] + ['optimization'.upper()]))
            step = lambda *args, **kwargs: \
                self.vanilla_step(*args, **kwargs) if self.optimization_step == 'vanilla' else self.adam_step(*args, **kwargs)
            while self.iter < self.max_iter and not self.vanishing_grad_flag:
                x = step(x=x, gradient_method=self.gradient_method)
                self.iter += 1
            if self.fine_tuning:
                print('Fine tuning with parameter shift rule')
                while self.iter < self.max_iter+5 and not self.vanishing_grad_flag:
                    x = step(x=x, gradient_method='parameter_shift')
                    self.iter += 1
        self.x = x
        self.time = time.time() - self.time
        return
