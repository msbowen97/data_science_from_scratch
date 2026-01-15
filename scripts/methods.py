import numpy as np 
import pandas as pd
import os, gc
import warnings
from scipy import stats
from joblib import Parallel, delayed, dump, load
from diagnostics import Diagnostics
import copy
import time

class Validator:
    def __init__(self, method, diagnosticParams={}, inlineDiagnostics=True, finalDiagnostics=True):
        self.methodPrototype = method
        self.diagnostics = Diagnostics(**diagnosticParams)
        self.inlineDiagnostics = inlineDiagnostics
        self.finalDiagnostics = finalDiagnostics

    def __call__(self, trainXs, trainYs, validationXs, validationYs, binary=True, threshold=0.5):
        self.methods = np.empty((len(trainXs),), dtype=object)
        self.diagnosticDicts = np.empty((len(trainXs),), dtype=object)
        for i in range(len(trainXs)):
            startTime = time.time()
            self.methods[i] = copy.deepcopy(self.methodPrototype)
            self.methods[i](trainXs[i], trainYs[i])
            self.methods[i].runtime = np.round(time.time() - startTime, 2)
            yHat = self.methods[i].predict(X=validationXs[i])

            if binary:
                yHat = (yHat > threshold) * 1

            self.diagnostics(y=validationYs[i], y_hat=yHat, output=self.inlineDiagnostics, runtime=self.methods[i].runtime)
            self.diagnosticDicts[i] = self.diagnostics.diagnostics.copy()

        if self.finalDiagnostics:
            self.ensembleDict, self.ensembleStr = self.diagnostics.calc_ensemble_diagnostics(self.diagnosticDicts)
            print(self.ensembleStr)

    def predict(self, X):
        yHats = np.empty((len(self.methods), X.shape[0]), dtype=np.float32)
        for i in range(len(self.methods)):
            yHats[i] = self.methods[i].predict(X)

        return np.mean(yHats, axis=0)
    
class Method:
    def __init__(self, modelMethod, lossMethod, optimizerMethod, modelParams={}, lossParams={}, optimizerParams={}):
        self.model = self._get_model(modelMethod=modelMethod, modelParams=modelParams)
        self.lossFunction = self._get_loss_function(modelMethod=modelMethod, lossMethod=lossMethod, lossParams=lossParams)
        self.optimizer = self._get_optimizer(optimizerMethod=optimizerMethod, optimizerParams=optimizerParams)

    def __call__(self, X, y):
        self.optimizer(X=X, y=y, model=self.model, lossFunc=self.lossFunction)

    def _get_model(self, modelMethod, modelParams):
        return Model(modelMethod = modelMethod, **modelParams)
    
    def _get_loss_function(self, modelMethod, lossMethod, lossParams):
        return LossFunction(modelMethod=modelMethod, lossMethod=lossMethod, **lossParams)
    
    def _get_optimizer(self, optimizerMethod, optimizerParams):
        optimizerAliases = {
            'gradient_descent': 'gd',
            'gd': 'gd',
            'minibatch_gradient_descent': 'gd',
            'minibatch_gd': 'gd',
            'mgd': 'gd',
            'stochastic_gradient_descent': 'gd',
            'stochastic_gd': 'gd',
            'sgd': 'gd',
            'simulated_annealing': 'sa',
            'sa': 'sa',
            'decision_tree': 'rf',
            'dt': 'rf',
            'random_forest': 'rf',
            'rf': 'rf',
            'gradient_booster': 'gb',
            'gb': 'gb'
        }

        optimizers = {
            'gd': GradientDescent,
            'sa': SimulatedAnnealing,
            'rf': RandomForest,
            'gb': GradientBooster
        }

        return optimizers[optimizerAliases[optimizerMethod.lower()]](optimizerMethod=optimizerMethod, **optimizerParams)
    
    def predict(self, X):
        return self.optimizer.predict(X)
        
class Model:
    '''
    Class to define and apply set machine learning models.

    Attributes
    ----------
    modelMethod : {'linreg', 'logreg', 'softmax'}
        Method to be used by this model. Defaults to 'linreg'.
    '''
    def __init__(self, modelMethod='linreg'):
        '''
        Initialize the Model.
        '''
        if not modelMethod or modelMethod not in ['linreg', 'logreg','softmax']:
            self.modelMethod = 'linreg'
            warnings.warn(f"Invalid or Nonetype model type provided. Defaulting to linreg.", UserWarning)
        else:
            self.modelMethod = modelMethod

    def _calc_y_hat(self, X, theta):
        '''
        Transform a user-provided inputs into model-based predictions.

        Parameters
        ----------
        X : np.array (n,m)
            Parameter space of the input data to be transformed.
        theta : np.array (m,)
            Weights to be applied to the feature space.
        '''
        methods = {
          'linreg': self._calc_lin_reg,
          'logreg': self._calc_log_reg,
          'softmax': self._calc_softmax
        }

        return methods[self.modelMethod](X, theta)
    
    def _calc_sigmoid(self, yHat):
        return 1 / (1 + np.exp(-yHat))

    def _calc_lin_reg(self, X, theta):
        '''
        Calculate the linear regression model output.
        '''
        return X @ theta
    
    def _calc_log_reg(self, X, theta):
        '''
        Calculate the logistic regression model output.
        '''
        yHat = self._calc_sigmoid(X @ theta)
        return yHat
    
    def _calc_softmax(self, X, theta):
        '''
        Calculate the softmax model output.
        '''
        z = self.lin_reg(X, theta)
        yHat = np.exp(z) / np.sum(np.exp(z), axis=0)
        return yHat
    
    def _calc_sigmoid(self, yHat):
        '''
        Apply the sigmoid function to the model output. Creates a probability space between 0 and 1.

        Parameters
        ----------
        yHat : np.array (n,)
            Array of output values generated by the model.
        '''
        return 1 / (1 + np.exp(-yHat))
  
class LossFunction:
    '''
    Class to define and apply set loss functions.

    Attributes
    ----------
    lossMethod : {'mse', 'mae', 'huber', 'binary_cross_entropy', 'cross_entropy', 'hinge'}
        Loss method to be used to evaluate the model. Defaults to 'mse'.
    modelMethod : {'linreg', 'logreg'}
        Model method to be used to calculate the loss gradient (where appropriate). Defaults to 'linreg'.
    yHat : np.array (n,)
        Model's predicted output. Generated during self.__call__(). 
    loss : numeric
        Average loss value of the model's prediction. Generated during self.__call__().
    grad : np.array (m,)
        Pointwise gradient of the model's prediction. Generated during self.__call__().
    '''
    def __init__(self, lossMethod='mse', modelMethod='linreg'):
        '''
        Initialize the LossFunction.
        '''
        self.lossMethod = lossMethod
        self.modelMethod = modelMethod

        if self.modelMethod == 'softmax' and self.lossMethod != 'cross_entropy':
            warnings.warn(f"Softmax must be run with cross_entropy. Setting loss_method to reflect this requirement.", UserWarning)
            self.lossMethod = 'cross_entropy'

    def __call__(self, X, y, yHat, calcGrad=True):
        '''
        Get the loss function and its gradient given model inputs and outputs.

        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        self.yHat = yHat
        self.loss = self.get_loss(y)
        self.grad = self.get_grad(X, y) if calcGrad else 0

    def get_loss(self, y):
        '''
        Calculate the appopriate loss function given the model predictions and actual output.

        Parameters
        ----------
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        losses = {
            'mse': lambda y: np.mean((y - self.yHat)**2),
            'mae': lambda y: np.mean(np.abs(y - self.yHat)),
            'huber': lambda y: np.mean(0.5 * (y - self.yHat) ** 2 if np.abs(y - self.yHat) <= self.delta \
                                                                    else self.delta * (np.abs(y - self.yHat) - self.delta / 2)),
            'binary_cross_entropy': lambda y: -np.mean(y * np.log(self.yHat+1e-10) + (1 - y) * np.log(1 - self.yHat+1e-10)),
            'cross_entropy': lambda y: np.mean(-y * np.ln(self.yHat)),
            'hinge': lambda y: np.max(0, 1 - y * self.yHat), # -1 <= y <= 1, used for SVMs
            'gini': lambda counts: 0.0 if counts.sum() == 0.0 else 1 - np.sum((counts / counts.sum()) ** 2),
            'entropy': lambda counts: 0.0 if counts.sum() == 0.0 else  -np.sum((counts / counts.sum()) * np.log2(counts / counts.sum() + 1e-10))
        }

        return losses[self.lossMethod](y)
    
    def get_grad(self, X, y):
        '''
        Calculate the appopriate pointwise gradient given the model predictions, inputs, and output.

        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        if self.modelMethod in ['softmax', 'logreg'] and self.lossMethod == 'cross_entropy':
            grad = X.T @ (self.yHat - y)
        else:
            dL_dy_hat = self.get_dL_dy_hat(y)
            dy_hat_dz = self.get_dy_hat_dz(X)
            dL_dz = dL_dy_hat * dy_hat_dz

            grad = X.T @ dL_dz

        return grad

    def get_dL_dy_hat(self, y):
        '''
        Calculate dL/dyHat portion of the larger gradient.

        Parameters
        ----------
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        dL_dy_hat = {
            'mse': lambda y: 2 * (self.yHat - y) / y.shape[0],
            'mae': lambda y: np.sign(self.yHat - y),
            'huber': lambda y: -(y - self.yHat) if np.abs(y - self.yHat) <= self.delta else -self.delta * np.sign(y - self.yHat),
            'binary_cross_entropy': lambda y: -(y / np.clip(self.yHat, 1e-10, 1 - 1e-10) - \
                                                (1 - y) / (1 - np.clip(self.yHat, 1e-10, 1 - 1e-10))) / y.shape[0],
            'cross_entropy': lambda y: -(y / self.yHat) / y.shape[0], # y must be one-hot encoded
            'hinge': lambda y: -y if y * self.y < 1 else 0,
        }

        return dL_dy_hat[self.lossMethod](y)

    def get_dy_hat_dz(self, X):
        '''
        Calculate dyHat/dz portion of the larger gradient.

        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        '''
        dy_hat_dz = {
            'linreg': lambda X: 1,
            'logreg': lambda X: self.yHat * (1 - self.yHat)
        }

        return dy_hat_dz[self.modelMethod](X)

class OptimizerPrototype:
    def __init__(self, optimizerMethod, maxEpochs=1e4, minEpochs=1e2, gradThreshold=1e-5, lossThreshold=1e-5, lossWindow=100, 
                 aliases = {}):
        self.optimizerMethod = optimizerMethod
        self.maxEpochs = maxEpochs
        self.minEpochs = minEpochs
        self.gradThreshold = gradThreshold
        self.lossThreshold = lossThreshold
        self.lossWindow = lossWindow
        self.losses, self.thetas = [], []
        self.aliases = aliases

    def __call__(self, X, y, model, lossFunc):
        self.model = model
        self.lossFunc = lossFunc
        self.colNames = X.columns
        self.X, self.y = self.shuffle(X=X.to_numpy(), y=y.to_numpy())

    def _generate_initial_theta(self, X, min=0.0, max=1.0):
        if self.initialGuess is None:
            warnings.warn(f"No initial guess provided for theta. Defaulting to a zero array.", UserWarning)
            return np.zeros(X.shape[1])
        elif self.initialGuess == 'zeros':
            return np.zeros(X.shape[1])
        elif self.initialGuess == 'random':
            return np.random.uniform(X.shape[1], low=min, high=max)
        else:
            return self.initialGuess

    def shuffle(self, X, y):
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    def train(self):
        pass

    def predict(self, X):
        return self.model._calc_y_hat(X=X, theta=self.theta)
    
    def _check_convergence(self, epoch, grad=[np.inf], losses=[np.inf]):
        '''
        Assess whether any of the convergence criteria have been met such that optimization should be halted.

        Parameters
        ----------
        grad : np.array
            List of loss function gradients appended after each optimization iteration.
        losses : np.array
            List of loss function averages appended after each optimization iteration.
        epoch : integer
            Current optimization iteration.
        '''
        if epoch >= self.minEpochs:
            if np.linalg.norm(grad) < self.gradThreshold:
                print(f'Descent halted after {epoch} epochs as gradient threshold reached ({np.round(np.linalg.norm(grad), 5)} <= {self.gradThreshold})')
                return True
            elif np.linalg.norm(np.diff(losses[-self.lossWindow:])) <= self.lossThreshold and len(losses) > self.lossWindow:
                print(f'Descent halted after {epoch} epochs as loss threshold reached ({np.round(np.linalg.norm(np.diff(losses[-self.lossWindow:])), 5)} <= {self.lossThreshold})')
                return True
            elif epoch == self.maxEpochs:
                print(f'Max epochs reached ({epoch})')
                return True
    
class GradientDescent(OptimizerPrototype):
    '''
    Class to define and apply a gradient descent optimizer.

    Attributes
    ----------
    modelMethod : {'linreg', 'logreg', 'softmax'}
        Model method to be optimized. Defaults to 'linreg'.
    optimizationMethod : {'gradient_descent', 'minibatch_gradient_descent', 'stochastic_gradient_descent', 'simulated_annealing', 
                          'decision_tree', 'random_forest', 'gradient_boosting'}
        Optimization method to be applied to the model. Defaults to 'gradient_descent'.
    lossMethod : {'mse', 'mae', 'huber', 'hinge', 'binary_cross_entropy', 'hinge'}
        Loss method to gauge accuracy and gradient of model predictions. Defaults to 'mse'.
    sigmoid : boolean
        Whether to apply the sigmoid function to the mode predictions and loss function.
    maxEpochs : integer
        Maximum number of iterations the optimization technique can be applied. Defaults to 1000.
    minEpochs : integer
        Minimum number of iterations the optimization technique must be applied. Defaults to 10.
    gradThreshold : numeric
        Minimum total gradient at which the optimization technique should continue. Defaults to 0.00001.
    lossThreshold : numeric
        Minimum average loss decrease at which the optimization technique should continue. Defaults to 0.
    lossWindow : integer
        Number of loss calculations to be used when evaluating the loss threshold. Defaults to 100. 
    thetas : np.array
        Running list of weights applied to the input data after each iteration.
    initialGuess : np.array (m,) OR {'zeros', 'random'}
        Initial guess for optimal theta. Defaults to np.zeros((m,)).
    losses : np.array
        Running list of losses calculated after each optimizatoin iteration.
    model : Model()
        Instance of the appropriately initialized Model() class.
    lossFunc : LossFunction()
        Instance of the appropriately initialized LossFunction() class.
    shuffledX : np.array (n,m)
        Feature data with shuffled rows. Same size as initial input data.
    shuffledY : np.array (n,)
        Output data with shuffled rows. Same permutation applied as with shuffledX.
    lr0 : numeric
        Initial learning rate to be used by the optimizer. Defaults to 0.1.
    lr : Updater()
        Learning rate currently being used by the optimizer.
    lrMin : numeric
        Minimum learning rate to be used by the optimizer. Defaults to 0.01.
    lrDecay : numeric
        Rate at which the learning rate decreases during optimization. Defaults to 0.0001.
    lrUpdate : {'percent', 'exponential', 'constant', 'linear'}
        Update method for the learning rate. Uses the Updater() class. Defaults to 'percent'.
    nBatch : integer
        Batch size to be used during batch gradient descent. Defaults to 64.
    momentumMethod : {None, 'nesterov', 'adamomentum'}
        Method to be used in calculating momentum during optimization. Defaults to None.
    momentumFactorFinal : numeric
        Ultimate momentum factor to be used in calculating momentum. Defaults to 0.9.
        Note: With momentumWarmup != 0, the momentum factor will begin at 0 and scale linearly.
    momentumWarmup : numeric
        Optimization iterations before reaching the final momentum factor. Defaults to 1000.
    momentumFactor : Updater()
        Momentum factor currently being used by the optimizer. Determined by momentumFactorFinal and momentumWarmup.
    theta : np.array (m,)
        Most recent guess at optimal weights to be applied to the input feature array.
    epochUpdate : integer (self.n / self.nBatch)
        Number of whole batches encompassed within the feature space. Set with _set_batch_params().
    '''
    def __init__(self, optimizerMethod, lr=0.1, lrMin=0.01, lrDecay=-1e-4, lrUpdate='percent', 
                 initialGuess='zeros', maxEpochs=1e4, minEpochs=10, gradThreshold=1e-5, lossThreshold=0, lossWindow=100,
                 nBatch=64, momentumMethod=False, momentumFactor=0.9, momentumWarmup=1000):
        '''
        Initialize the GradientDescent class.
        '''
        aliases = {
            'gradient_descent': 'gd',
            'gd': 'gd',
            'minibatch_gradient_descent': 'mgd',
            'minibatch_gd': 'mgd',
            'mgd': 'mgd',
            'stochastic_gradient_descent': 'sgd',
            'stochastic_gd': 'sgd',
            'sgd': 'sgd',
        }
        super().__init__(optimizerMethod=optimizerMethod, maxEpochs=maxEpochs, minEpochs=minEpochs, gradThreshold=gradThreshold,
                         lossThreshold=lossThreshold, lossWindow=lossWindow, aliases=aliases)
        
        self.lr0 = lr
        self.lrMin = lrMin
        self.lrDecay = lrDecay
        self.lrUpdate = lrUpdate
        self.nBatch = nBatch
        self.momentumMethod = momentumMethod
        self.momentumFactorFinal = momentumFactor
        self.momentumWarmup = momentumWarmup
        self.initialGuess = initialGuess

    def __call__(self, X, y, model, lossFunc):
        '''
        Optimize a feature array using the initialized gradient descent algorithm.

        Parameters
        ----------
        X : pd.DataFrame (n,m)
            Feature space of the input data.
        y : pd.DataFrame (n,)
            Output data being predicted by yHat.
        model : Model
            Model class used to predict yHat.
        lossFunc : LossFunction 
            LossFunction class used to calculate theta loss and gradients.
        '''
        super().__call__(X=X, y=y, model=model, lossFunc=lossFunc)
        self.theta = self._generate_initial_theta(X)
        self.thetas.append(self.theta)
        self.lossFunc(X=self.X, y=self.y, yHat=self.predict(X=self.X))
        self.losses.append(self.lossFunc.loss)
        self.lr = Updater(val=self.lr0, updateMethod=self.lrUpdate, rate=self.lrDecay, minVal=self.lrMin)
        self.momentumFactor = Updater(val=0, updateMethod='linear', rate=self.momentumFactorFinal/self.momentumWarmup, \
                                      maxVal=self.momentumFactorFinal)
        
        self.train()

    def _set_batch_params(self, X):
        '''
        Initialize the batch gradient descent parameters.
        Note: Called for all types of gradient descent. Sets epochUpdate and nBatch if appropriate.

        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        '''
        if self.aliases[self.optimizerMethod] == 'gd':
            self.epochUpdate = 1
            self.nBatch = X.shape[0]
        elif self.aliases[self.optimizerMethod] == 'mgd':
            self.nBatch = self.nBatch if self.nBatch else 64
            self.epochUpdate = int(X.shape[0] / self.nBatch)
        elif self.aliases[self.optimizerMethod] == 'sgd':
            self.nBatch = 1
            self.epochUpdate = X.shape[0]

    def _get_minibatch(self):
        '''
        Update the subset of the feature space currently being used to train the model.

        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        epoch : integer
            Current iteration of the gradient descent optimization.
        '''
        start, end = int((self.epoch.val%self.epochUpdate)*self.nBatch), int((self.epoch.val%self.epochUpdate + 1)*self.nBatch)
        return self.X[start:end], self.y[start:end]
    
    def _get_XY(self):
        '''
        Retrieve subset of feature space given descent parameters.

        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        epoch : integer
            Current iteration of the gradient descent optimization.
        '''

        Xs = [self.X[i*self.nBatch : (i+1)*self.nBatch] for i in range(self.epochUpdate)]
        Ys = [self.y[i*self.nBatch : (i+1)*self.nBatch] for i in range(self.epochUpdate)]

        # if self.aliases[self.optimizerMethod] == 'gd':
        #     return self.X, self.y
        # else:
        #     if batchNo == 0:
        #         self.X, self.y = self.shuffle(self.X,self.y)
        #     miniX, miniY = self._get_minibatch()
        
        return Xs, Ys

    def _update_theta(self, v, grad):
        '''
        Update theta given model parameters.

        Parameters
        ----------
        epoch : integer
            Current iteration of the gradient descent optimization.
        v : numeric
            ???
        grad : np.array (m,)
            Current gradient of feature space and theta.
        '''
        if not self.momentumMethod:
            theta = self.theta - self.lr.val * grad
        elif self.momentumMethod == 'nesterov':
            v_next = self.momentumFactor.val * v - self.lr.val * grad
            theta = self.theta - self.momentumFactor.val * v + (1 + self.momentumFactor.val) * v_next
            v = v_next
        elif self.momentumMethod == 'adamomentum': # Adamomentum
            bias = 1 / (1 - self.momentumFactor.val ** (self.epoch.val+1))
            v = self.momentumFactor.val * v + (1 - self.momentumFactor.val) * grad
            theta = self.theta - self.lr.val * v * bias
        
        self.thetas.append(theta)
        return theta, v

    def train(self):
        '''
        Iterate over the initialized gradient descent algorithm.
        '''
        v = np.zeros_like(self.theta) if self.momentumMethod else None
        self.epoch = Updater(0, updateMethod='linear', rate=1, maxVal=self.maxEpochs)
        self._set_batch_params(self.X)

        while True:
            self.lr.update()
            epochLoss, epochGrad = 0, np.zeros((self.X.shape[1]))

            if self.aliases[self.optimizerMethod] in ['mgd', 'sgd'] or self.epoch.val == 0:
                self.shuffle(self.X, self.y)

            xBatches, yBatches = self._get_XY()

            for i in range(self.epochUpdate):
                batchX, batchY = xBatches[i], yBatches[i]

                yHat = self.predict(batchX)
                self.lossFunc(X=batchX, y=batchY, yHat=yHat)

                epochLoss += self.lossFunc.loss
                grad = self.lossFunc.grad
                epochGrad += grad

                self.theta, v = self._update_theta(grad=grad, v=v)
            
            self.momentumFactor.update()
            self.losses.append(epochLoss / self.epochUpdate)
            self.epoch.update()

            if self._check_convergence(grad=epochGrad/self.epochUpdate, losses=self.losses, epoch=self.epoch.val):
                break
            
class SimulatedAnnealing(OptimizerPrototype):
    '''
    Class for defining and applying a simulated annealing optimization algorithm.

    Attributes
    ----------
    {Copy from above}
    stepSize : numeric
        Magnitude of changes available for each adjustment to theta. Defaults to 10.
    temperature : Updater()
        Value determining the probability a suboptimal guess will be further explored. Defaults to 1000.
    cooling : {'linear', 'exponential', 'percent', 'const'}
        Update method for the temperature value. Defaults to 'exponential'.
    coolingRate : numeric
        Rate at which the temperature is adjusted. Defaults to 0.00001.
    loss : numeric
        Value of the loss function given the current optimal theta.
    bestTheta : np.array (m,)
        Theta resulting in the lowest value of the loss function. 
    bestLoss : numeric
        Loss function value attributed to bestTheta.
    '''
    def __init__(self, optimizerMethod, temperature=1e3, cooling='exponential', coolingRate=-1e-5,
                 stepSize = 10, initialGuess='zeros', maxEpochs=1e5, minEpochs=10, gradThreshold=0, lossThreshold=1e-5, 
                 lossWindow=100):
        '''
        Initialize the SimulatedAnnealing class.
        '''
        aliases = {
            'simulated_annealing': 'sa',
            'sa': 'sa'
        }
        super().__init__(optimizerMethod=optimizerMethod, maxEpochs=maxEpochs,
                         minEpochs=minEpochs, gradThreshold=gradThreshold, lossThreshold=lossThreshold,
                         lossWindow=lossWindow, aliases=aliases)
        self.initialGuess = initialGuess
        self.stepSize = stepSize
        self.temperature = Updater(val=temperature, updateMethod=cooling, rate=coolingRate, minVal=0, maxVal=temperature)
        self.thetas, self.losses = [], []

    def __call__(self, X, y, model, lossFunc):
        '''
        Run the initialized simulated annealing optimization algorithm.
        
        
        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        super().__call__(X=X, y=y, model=model, lossFunc=lossFunc)
        self.theta = self._generate_initial_theta(X)
        self.train()

    def _alter_theta(self):
        '''
        Apply a random perturbation to theta
        '''
        index = np.random.randint(0,len(self.theta)-1)
        self.theta[index] += np.random.uniform(-self.stepSize, self.stepSize)

    def train(self):
        '''
        Iterate over the initialized simulated annealing optimization algorithm.
        
        Parameters
        ----------
        X : np.array (n,m)
            Feature space of the input data.
        y : np.array (n,)
            Output data being predicted by yHat.
        '''
        self.bestTheta = self.theta.copy()
        self.thetas.append(self.theta.copy())
        
        yHat = self.predict(X=self.X)
        self.lossFunc(X=self.X, y=self.y, yHat=yHat)
        self.bestLoss = self.lossFunc.loss.copy()
        self.losses.append(self.lossFunc.loss)

        self.epoch = Updater(0, updateMethod='linear', rate=1, maxVal=self.maxEpochs)
        while True:
            self._alter_theta()
            yHat = self.predict(X=self.X)
            self.lossFunc(X=self.X, y=self.y, yHat=yHat)
            self.loss = self.lossFunc.loss

            self.thetas.append(self.theta.copy())
            self.losses.append(self.loss)

            if self.loss < self.bestLoss:
                self.bestTheta = self.theta.copy()
                self.bestLoss = self.loss.copy()
            elif np.random.random() <= np.exp(-(self.loss - self.bestLoss) / self.temperature.val):
                pass 
            else:
                self.theta = self.bestTheta.copy()

            self.epoch.update()
            self.temperature.update()

            if self._check_convergence(losses=self.losses, epoch=self.epoch.val):
                break
            
class DecisionTree(OptimizerPrototype): # Only for classification ATM
    '''
    Class for defining and applying a decision tree optimization algorithm.

    Attributes
    ----------
    {see above}
    maxTreeDepth : integer
        Maximum number of splits the algorithm will allow.
    minGroupSize : integer
        Minimum number of samples required on each side of a valid split.
    nCols : percent
        Percentage of columns to be exposed at each node. Defaults to 1 (all columns).
    nRows : percent
        Percentage of samples to be exposed at each node. Defaults to 1 (all rows).
    splitMethod : {'full_sweep', 'histogram'}
        Method for identifying valid splits within the feature space.
    colNames : np.array (m,)
        Names of columns associated with the feature space to be explored.
    currentDepth : integer
        Number of parent nodes preceding the current split. Defaults to 0.
    bestCol : integer
        Index of the column providing the optimial split in subset of overall feature space. Set with _calc_impurities().
    splitColIndex : integer
        Index of the column providing the optimial split in overall feature space. Set with _update_params().
    splitCol : str
        Name of the column providing the optimal split in the overall feature space. Set with _update_params().
    splitPt : numeric
        Value of the optimial split point in the overall feature space. Set with _calc_impurities().
    prediction : numeric
        Predicted output of a sample occupying the current node. Set with _get_prediction().
    nClasses : integer
        Number of unique values to be predicted during classification. Set during initial __call__().
    binNo : integer
        Number of bins used in current histogram calculations. Updates based on size of feature space.
    prevBinNo : integer
        Number of bins used in the previous histogram.
    classification : boolean
        Whether to perform classification (True) or regression (False). Defaults to True.
    thetas : np.empty (0,)
        Placeholder to feed functions requiring a thetas list.
    theta : None
        Placeholder to feed functions requiring a theta array.
    left : np.array (nL,m)
        Node containing samples with features beneath the optimal split point.
    right : np.array (nR,m)
        Node containing samples with features above the optimal split point.
    bins : np.array (m,4-16)
        Percentiles associated with the current binNo for each feature in the current feature space. Set by _get_bins().
    binIds : np.array (m,4-16)
        Ordinal indices associated with the bins of each feature in the current feature space. Set by _get_splits().
    branchCols : np.array (int(m*nCol),)
        Indices of columns to be evaluated for splits. Randomly selected by _col_select().
    branchRows : np.array (int(n*nRows),)
        Indices of rows to be used in evaluating splits. Randomly selected by _row_select().
    '''
    def __init__(self, optimizerMethod='decision_tree', lossMethod='gini', maxTreeDepth=3, minGroupSize=5, nCols=1, nRows=1, splitMethod='full_sweep', 
                 colNames = None, currentDepth=0, splitCol=None, splitPt=None, prediction=None, nClasses=None,
                 prevBinNo = 0, classification=True):
        aliases = {
            'decision_tree':'dt',
            'dt': 'dt',
            'random_forest':'rf',
            'rf': 'rf'
        }
        super().__init__(optimizerMethod=optimizerMethod, aliases=aliases)
        self.lossMethod = lossMethod
        self.maxTreeDepth = maxTreeDepth
        self.currentDepth = currentDepth
        self.minGroupSize = minGroupSize
        self.splitCol, self.splitPt, self.prediction = splitCol, splitPt, prediction
        self.nCols, self.nRows = nCols, nRows
        self.thetas = [None]
        self.colNames = colNames
        self.nClasses = nClasses
        self.left, self.right = None, None
        self.splitMethod = splitMethod
        self.prevBinNo = prevBinNo
        self.classification = classification
        self.bestCol = None
        self.theta = None

        if not self.classification and self.lossMethod in ['gini', 'entropy']:
            print(f'{self.lossMethod} is incompatible with regression analysis. Setting loss methohd to `mse`.')
            self.lossMethod = 'mse'

    def __call__(self, X, y, model, lossFunc, binIds=[], bins=[]):
        self.lossFunc = lossFunc
        self.X, self.y = self.shuffle(X=X, y=y)
        self._get_prediction(self.y)
        self.n = y.shape[0]
        self.binNo = min(16, max(4, int(np.sqrt(self.n))))
        self.binIds = binIds
        self.bins = bins
        if self.currentDepth == 0:
            if self.colNames is None:
                self.colNames == X.columns
                X, y = X.to_numpy(), y.to_numpy().astype(np.int64)

            self.nClasses = len(np.unique(y))

            if self.splitMethod == 'histogram':
                self.bins = np.empty(X.shape[1], dtype=object)
                self._get_bins(X)
                self._get_splits(X)

        elif self.splitMethod == 'histogram' and self.prevBinNo != self.binNo:
            self.bins = np.empty(X.shape[1], dtype=object)
            self._get_bins(X)
            self._get_splits(X)
            
        if self.currentDepth < self.maxTreeDepth:
            self._create_decision_tree(X, y)

    def _get_bins(self, X):
        for i in range(X.shape[1]):
            self.bins[i] = np.unique(np.percentile(X[:,i], np.linspace(0, 100, self.binNo+1)))

    def _get_splits(self, X):
        self.binIds = np.zeros((X.shape[1], X.shape[0]), dtype=np.int32)
        for col in range(X.shape[1]):
            bins = self.bins[col]
            self.binIds[col] = np.searchsorted(bins, X[:,col], side="left") - 1
            self.binIds[col] = np.clip(self.binIds[col], 0, max(0,len(bins)-2))
    
    def _col_select(self, cols):
        self.branchCols = np.random.choice(cols, size=int(self.nCols * cols), replace=False)
    
    def _row_select(self, rows):
        self.branchRows = np.random.choice(rows, size=int(self.nRows * rows), replace=True)

    def _calc_impurity(self, yL, yR, nL, nR):
        if self.classification:
            child1Impurity = self.lossFunc.get_loss(y=yL)
            child2Impurity = self.lossFunc.get_loss(y=yR)
        else:
            yHatL, yHatR = np.mean(yL), np.mean(yR)
            self.lossFunc(X=None, y=yL, yHat=yHatL, calcGrad=False)
            child1Impurity = self.lossFunc.loss

            self.lossFunc(X=None, y=yR, yHat=yHatR, calcGrad=False)
            child2Impurity = self.lossFunc.loss

        N = nR + nL

        return (nL / N) * child1Impurity + (nR / N) * child2Impurity

    def _calc_full_sweep(self, X, y):
        N = y.shape[0]
        bestResult = np.inf

        for col in range(X.shape[1]):
            order = np.argsort(X[:,col])
            Xsorted = X[:,col][order]
            ySorted = y[order]

            if self.classification:
                countsL = np.zeros(self.nClasses, dtype=np.int32)
                countsR = np.array([np.sum(y==i) for i in np.unique(y)])
            else:
                pass

            for i in range(1,N):
                if self.classification:
                    cls = ySorted[i-1]
                    countsL[cls] += 1
                    countsR[cls] -= 1
                else:
                    pass

                if Xsorted[i-1] == Xsorted[i]:
                    continue
                
                if i < self.minGroupSize or N - i < self.minGroupSize:
                    continue
                
                if self.classification:
                    impurity = self._calc_impurity(yL=countsL, yR=countsR, nL=i, nR=N-i)
                else:
                    impurity = self._calc_impurity(yL=ySorted[:i], yR=ySorted[i:], nL=i, nR=N-i)

                if impurity < bestResult:
                    bestResult = impurity
                    self.splitPt = (Xsorted[i-1] + Xsorted[i]) / 2
                    self.bestCol = col

    def _calc_histogram(self, X, y):
        N = y.shape[0]
        bestResult = np.inf

        self.branchBinIds = self.binIds[self.branchCols][:,self.branchRows]

        for col in range(X.shape[1]):
            colIds = self.branchBinIds[col]
            if self.classification:
                hist = np.zeros((self.binNo, self.nClasses), dtype=np.int32)

                flat = colIds * self.nClasses + y
                hist = np.bincount(flat, minlength=self.binNo * self.nClasses).reshape(self.binNo, self.nClasses)

                lHist, nL = np.zeros(self.nClasses, dtype=np.int32), 0
                rHist, nR = hist.sum(axis=0).astype(np.int32), N
            else:
                mask, nL, nR = np.zeros(N, dtype=int), 0, N

            for bin in range(self.binNo-1):
                if self.classification:
                    counts = hist[bin]
                    lHist += counts
                    rHist -= counts

                    delta = counts.sum()
                    nL += delta
                    nR -= delta
                else: 
                    mask += colIds == bin
                    nL = np.sum(mask)
                    nR = N - nL

                if nL < self.minGroupSize or nR < self.minGroupSize:
                    continue
                
                if self.classification:
                    impurity = self._calc_impurity(yL=lHist, yR=rHist, nL=nL, nR=nR)
                else:
                    impurity = self._calc_impurity(yL=y[mask], yR=y[~mask], nL=nL, nR=nR)

                if impurity < bestResult:
                    bestResult = impurity
                    self.splitPt = (self.bins[self.branchCols[col]][bin] + self.bins[self.branchCols[col]][bin+1])/2
                    self.bestCol = col
    
    def _calc_impurities(self, X, y):
        if self.classification:
            self.parentImpurity = self.lossFunc.get_loss(y=np.bincount(y, minlength=self.nClasses))
        else:
            self.lossFunc(X=X, y=y, yHat=np.mean(y), calcGrad=False)
            self.parentImpurity = self.lossFunc.loss

        if self.parentImpurity <= 1e-5:
            self.splitPt = None
            return

        self.bestSplit = 0

        if self.splitMethod == 'full_sweep':
            self._calc_full_sweep(X=X, y=y)

        elif self.splitMethod == 'histogram':
            self._calc_histogram(X=X, y=y)

    def _update_params(self, X):
        self.splitColIndex = self.branchCols[self.bestCol]
        self.splitCol = self.colNames[self.splitColIndex]

    def _create_branches(self, X, y):
        mask = (X[:,self.branchCols[self.bestCol]] <= self.splitPt).astype(bool)

        nL = mask.sum()
        nR = self.n - nL

        if nL < self.minGroupSize or nR < self.minGroupSize:
            return
        
        self.left = DecisionTree(optimizerMethod=self.optimizerMethod, maxTreeDepth=self.maxTreeDepth, 
                                 currentDepth=self.currentDepth+1, minGroupSize=self.minGroupSize, colNames=self.colNames,
                                 nClasses=self.nClasses, splitMethod=self.splitMethod, prevBinNo=self.binNo, 
                                 classification=self.classification, lossMethod=self.lossMethod)
        
        lBinIds = [] if self.binIds == [] else self.binIds[:,mask]
        self.left(X[mask], y[mask], model=None, lossFunc=self.lossFunc, binIds=lBinIds, bins=self.bins)

        self.right = DecisionTree(optimizerMethod=self.optimizerMethod, maxTreeDepth=self.maxTreeDepth, 
                                 currentDepth=self.currentDepth+1, minGroupSize=self.minGroupSize, colNames=self.colNames,
                                 nClasses=self.nClasses, splitMethod=self.splitMethod, prevBinNo=self.binNo,
                                 classification=self.classification, lossMethod=self.lossMethod)
        rBinIds = [] if self.binIds == [] else self.binIds[:,~mask]
        self.right(X[~mask], y[~mask], model=None, lossFunc=self.lossFunc, binIds=rBinIds, bins=self.bins)

    def _create_decision_tree(self, X, y):
        self._col_select(X.shape[1])
        self._row_select(X.shape[0])

        # self.splitCols, self.splitPts = self._get_splits(X[self.branchRows][:, self.branchCols])
        self._calc_impurities(X[self.branchRows][:, self.branchCols], y[self.branchRows])
        if self.bestCol is not None:
            self._update_params(X=X)
            self._create_branches(X=X, y=y)
        else:
            pass

    def _get_prediction(self, y):
        self.prediction = np.mean(y)

    def predict(self, X, theta=None, binary=False):
        n = X.shape[0]
        y_pred = np.empty(n, dtype=np.float64)

        stack = [(self, np.arange(n))]

        while stack:
            branch, idx = stack.pop()

            if not branch.left:
                y_pred[idx] = branch.prediction
                continue

            try: # Fix later, assume numpy, update DecisionTree
                col = X.iloc[idx, branch.splitColIndex]
                mask = col <= branch.splitPt

                left_idx = idx[mask.values]
                right_idx = idx[~mask.values]

            except:
                col = X[idx, branch.splitColIndex]
                mask = col <= branch.splitPt

                left_idx = idx[mask]
                right_idx = idx[~mask]

            stack.append((branch.left, left_idx))
            stack.append((branch.right, right_idx))

        if binary:
            return np.round(y_pred)
        else:
            return y_pred
        
class RandomForest(DecisionTree):
    def __init__(self, optimizerMethod='rf', lossMethod='gini', nTrees=100, nCols=0.2, nRows=0.5, maxTreeDepth=10, minGroupSize=5, 
                 splitMethod='histogram', colNames=None, classification=True):
        super().__init__(optimizerMethod=optimizerMethod, maxTreeDepth=maxTreeDepth, minGroupSize=minGroupSize,
                         splitMethod=splitMethod, colNames=colNames, lossMethod=lossMethod, classification=classification)
        self.nTrees = nTrees if self.optimizerMethod != 'decision_tree' else 1
        self.nCols = nCols if self.optimizerMethod != 'decision_tree' else 1
        self.nRows = nRows if self.optimizerMethod != 'decision_tree' else 1
        self.theta = None

    def __call__(self, X, y, model, lossFunc):
        self.lossFunc = lossFunc
        self.train(X, y)

    def _create_tree(self, X, y, seed):
        np.random.seed(seed)
        tree = DecisionTree(optimizerMethod=self.optimizerMethod, lossMethod=self.lossMethod, maxTreeDepth=self.maxTreeDepth, 
                                minGroupSize=self.minGroupSize, nCols=self.nCols, nRows=self.nRows, splitMethod=self.splitMethod,
                                colNames=self.colNames, classification=self.classification)
        tree(X,y,model=None,lossFunc=self.lossFunc)

        return tree
    
    def _create_mmap(self, X, y):
        base = os.path.abspath("mmap")
        os.makedirs(base, exist_ok=True)

        Xpath = os.path.join(base, "X.joblib")
        yPath = os.path.join(base, "y.joblib")
        Xtmp = os.path.join(base, "X.tmp.joblib")
        yTmp = os.path.join(base, "y.tmp.joblib")

        for name in ("Xshared", "yShared"):
            if name in globals():
                del globals()[name]
        gc.collect()

        dump(np.ascontiguousarray(X), Xtmp, compress=0)
        dump(np.ascontiguousarray(y), yTmp, compress=0)

        # Atomic replace 
        os.replace(Xtmp, Xpath)
        os.replace(yTmp, yPath)

        dump(X, os.path.abspath("mmap/X.joblib"), compress=0)
        dump(y, os.path.abspath("mmap/y.joblib"), compress=0)

        return load(Xpath, mmap_mode="r"), load(yPath, mmap_mode="r")

    def train(self, X, y):
        self.colNames = X.columns if self.colNames is None else self.colNames

        if self.classification:
            X_np, y_np = np.asarray(X, dtype=np.float32, order="C"), np.asarray(y, dtype=np.int32, order="C")
        else:
            X_np, y_np = np.asarray(X, dtype=np.float32, order="C"), np.asarray(y, dtype=np.float64, order="C")
        Xshared, yShared = self._create_mmap(X=X_np, y=y_np)

        if self.nTrees > 5:
            self.forest = Parallel(n_jobs=4, backend='loky')(delayed(self._create_tree)(Xshared, yShared, seed) 
                                                             for seed in range(self.nTrees))
        else:
            self.forest = [self._create_tree(Xshared, yShared, seed) for seed in range(self.nTrees)]

    def predict(self, X, theta=None, binary=False):
        yHat = np.zeros((X.shape[0],len(self.forest)))
        for i in range(len(self.forest)):
            yHat[:,i] = self.forest[i].predict(X, binary=False)

        if binary:
            return stats.mode(yHat, axis=1).mode[:, 0]
        else:
            return np.mean(yHat, axis=1)
        
class GradientBooster(RandomForest): # Flawed, gradients not matching loss function, need to incorporate Hessian?
    def __init__(self, optimizerMethod, lossMethod, nCols=1, nRows=1, maxTreeDepth=2, minGroupSize=5, splitMethod='histogram',
                 maxBoosterDepth=200, lr=0.1, classification=False):
        super().__init__(lossMethod='mse', maxTreeDepth=maxTreeDepth, minGroupSize=minGroupSize,
                         splitMethod=splitMethod, nCols=nCols, nRows=nRows, nTrees=1, classification=False)
        self.maxBoosterDepth = maxBoosterDepth
        self.lr = lr
        self.yProb = None

    def __call__(self, X, y, model, lossFunc):
        self.lossFunc = lossFunc
        self.colNames = X.columns
        X_np, y_np = np.asarray(X, dtype=np.float32, order="C"), np.asarray(y, dtype=np.int32, order="C")
        self.train(X_np, y_np)

    def _calc_sigmoid(self, yHat):
        z = np.clip(yHat, -20, 20)
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y):
        N = y.shape[0]
        self.targets, self.boosters = np.zeros((self.maxBoosterDepth, N), dtype=np.float64), np.empty((self.maxBoosterDepth,), dtype=object)
        self.targets[0] = y
        treePrototype = RandomForest(lossMethod=self.lossMethod, maxTreeDepth=self.maxTreeDepth, nTrees=1,
                                    minGroupSize=self.minGroupSize, nCols=1, nRows=1, splitMethod=self.splitMethod,
                                    colNames=self.colNames, classification=self.classification)
        
        yMean = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        yHat = np.full(N, np.log(yMean / (1 - yMean)))

        for i in range(self.maxBoosterDepth):
            prob = self._calc_sigmoid(yHat)
            grad = y - prob
            self.boosters[i] = copy.deepcopy(treePrototype)
            self.boosters[i](X, grad, model=None, lossFunc=self.lossFunc)
            yHat += self.boosters[i].predict(X=X, theta=self.theta, binary=False) * self.lr

            print(
                f"iter {i}: "
                f"prob mean={prob.mean():.3f}, "
                f"grad std={grad.std():.3f}, "
                f"yHat std={yHat.std():.3f}"
            )

    def predict(self, X, theta=None, binary=True):
        yHat = np.zeros((X.shape[0],), dtype=np.float64)
        for i in range(len(self.boosters)):
            yHat += self.boosters[i].predict(X=X, theta=theta, binary=False) * self.lr

        if binary:
            self.yProb = self._calc_sigmoid(yHat=yHat)
            return np.round(self.yProb) # assumes turning point at 0.5
        else:
            return yHat
      
    # def newton(self, X, y):
    #     pass

    # def genetic_algorithm(self, X, y):
    #     pass

    # def bayesian(self, X, y):
    #     pass

class Updater:
    '''
    Class to iteratively update variables.

    Attributes
    ----------
    val : numeric, optional
        Current value of the variable. Updates at each .update() call. Defaults to 0 on initialization.
    val0 : numeric, optional
        Initial value assigned to the variable. Assigned same value as self.val on initialization.
    updateMethod : {'linear', 'exponential', 'const', 'percent'}, optional
        Method for updating the variable at each iteration. Defaults to 'linear'.
    rate : numeric
        Rate at which the variable grows/shrinks. Defaults to 1.
    minVal : numeric, optional
        Minimum value the variable is allowed to reach. Defaults to 0.
    maxVal : numeric, optional
        Maximum value the variable is allowed to reach. Defaults to np.inf.
    iteration : integer
        Number of times the variable has been updated.
    '''
    def __init__(self, val=0, updateMethod='linear', rate=1, minVal=0, maxVal=np.inf):
        '''
        Initialize the Updater.
        '''
        self.val, self.val0 = val, val
        self.updateMethod = updateMethod
        self.rate = rate
        self.minVal = minVal
        self.maxVal = maxVal
        self.iteration = 0

    def update(self):
        '''
        Update the variable using the appropriate method.
        '''
        methods = {
            'linear': self._calc_linear,
            'exponential': self._calc_exp_growth,
            'const': self._calc_const,
            'percent': self._calc_percent_growth
        }
        
        self.val = methods[self.updateMethod]()
        self.iteration += 1

    def _calc_linear(self):
        '''
        Add one rate step to the current value.
        '''
        return np.clip(self.val + self.rate, self.minVal, self.maxVal)
    
    def _calc_exp_growth(self):
        '''
        Calculate the exponential growth of the variable at the set rate and iterations.
        '''
        return np.clip(self.val0 * np.exp(self.rate * self.iteration), self.minVal, self.maxVal)
                       
    def _calc_percent_growth(self):
        '''
        Calculate the percent growth of the variable at the set rate and iterations.
        '''
        return np.clip(self.val0 * (1 + self.rate) ** self.iteration, self.minVal, self.maxVal)
    
    def _calc_const(self):
        '''
        Return the current value of the variable.
        '''
        return self.val