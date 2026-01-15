import numpy as np
import pandas as pd
import warnings
import methods
import time
from diagnostics import Diagnostics
import copy

class Preprocessor:
    def __init__(self, onehotCols=None, directCols=None, ordinalCols=None, dropCols=None, imputerMethod='lazy', normalizerMethod='zscore',
                 splitterMethod='kfold', validationSplit=0.2, splitGroups=5, bagging=True):
        self.quantifier = Quantifier(onehotCols=onehotCols, directCols=directCols, ordinalCols=ordinalCols, dropCols=dropCols)
        self.imputer = Imputer(method=imputerMethod)
        self.normalizer = Normalizer(method=normalizerMethod)
        self.splitter = Splitter(method=splitterMethod, validationSplit=validationSplit, groups=splitGroups, bagging=bagging)

    def __call__(self, X, y):
        X = self.transform(X)
        self.split(X, y)

    def transform(self, X, means=None, stds=None, maximum=None, minimum=None, norms=None):
        return self.normalizer(self.imputer(self.quantifier(X)), maximum=maximum, minimum=minimum, means=means, stds=stds,
                               norms=norms)
    
    def split(self, X, y):
        self.splitter(X,y)
    
class Quantifier:
    def __init__(self, onehotCols=None, directCols=None, ordinalCols=None, dropCols=None):
        self.onehotCols = onehotCols
        self.directCols = directCols
        self.ordinalCols = ordinalCols
        self.dropCols = dropCols

    def _check_numeric(self, X):
        '''
        Identify numeric (True) and non-numeric (False) columns.

        Returns
        -------
        normCols: ndarray of shape (n_cols,)
            Boolean array of numeric (True) and non-numeric (False) columns.
        '''
        return np.array([pd.api.types.is_numeric_dtype(X[col]) for col in X.columns])

    def __call__(self, X):
        X = X.drop(self.dropCols, axis=1)
        X = self.direct_encode(X.copy())
        X = self.ordinal_encode(X.copy())
        subX = self.onehot_encode(X[self.onehotCols].copy())

        quantifiedX = pd.concat([X.drop(self.onehotCols, axis = 1), subX], axis=1)
        numCols = self._check_numeric(quantifiedX)

        if np.mean(numCols) < 1:
            print('Dropping non-numeric columns: {quantifiedX.columns[~numCols]}')
            quantifiedX.drop(quantifiedX.columns[~numCols], inplace=True)

        return quantifiedX

    def direct_encode(self, X): 
        for val in self.directCols:
            X[val] = pd.to_numeric(X[val], errors='coerce')
        return X
    
    def ordinal_encode(self, X):
        for key, valList in self.ordinalCols.items():
            for val in valList:
                X.loc[X[key] == val, key] = self.ordinalCols[key].index(val)
            X[key] = X[key].astype(float)

        return X

    def onehot_encode(self, subX):
        for col in self.onehotCols:
            distinctValues = np.unique(subX[col].dropna())  # only real values

            for val in distinctValues[:-1]:
                subX.loc[:, f'{col}_{val}'] = subX[col].apply(
                    lambda x: 1 if x == val else (0 if pd.notna(x) else None)
                )
            # drop original column
            subX = subX.drop(col, axis=1)

        return subX
    
class Imputer(Preprocessor):
    '''
    Class to impute missing values into a feature space.
    '''
    def __init__(self, method):
        self.method = method

        if self.method not in ['lazy']:
            warnings.warn(f"Invalid or Nonetype method ({self.method}); defaulting to `lazy`.", UserWarning)
            self.method = 'lazy'

    def __call__(self, X):
        methods = {
            'lazy' : lambda X : self.lazy_impute(X),
            'iterative' : lambda X : self.iterative(X)
                }
        
        return methods[self.method](X)
        
    def lazy_impute(self, X):
        for col in X.columns:
            if np.sum(X[col].isna()) > 0:
                X[col + "_missing"] = X[col].isna().astype(float)
            X[col] = X[col].astype(float).fillna(X[col].median())

        return X
    
    def iterative_imputer(self, X):
        pass

class Normalizer:
    '''
    Class to normalize the numeric data of a feature space.

    This class normalizes numeric data based on a user-chosen method.

    Attributes
    ----------
    method : {'zscore', 'minmax', 'log', 'l1', 'l2'}, optional
        User-chosen method of normalization. Default is 'minmax'.
    '''

    def __init__(self, method='zscore'):
        '''
        Initialize the Normalizer.
        '''
        self.method = method

        if self.method not in ['zscore', 'minmax', 'log', 'l1', 'l2']:
            warnings.warn(f"Invalid or Nonetype method ({method}); defaulting to zscore.", UserWarning)
            self.method = 'zscore'
    
    def __call__(self, X, minimum=0, maximum=1, means=None, stds=None, norms=None):
        '''
        Transform the numeric data using a user-chosen method.

        Parameters
        ----------
        X : pandas.Dataframe
            Dataframe of featue space to normalize.
        minimum: float, optional
            Minimum value desired for 'minmax' normalization. Default is 0.
        maximum: float, optional
            Maximum value desired for 'minmax' normalization. Default is 1.

        Returns
        -------
        pandas.DataFrame
            Normalized feature space with non-numeric columns left unchanged.
        '''

        methods = {
            'zscore': lambda X: self.zscore(X, means, stds),
            'log':    lambda X: self.log(X),
            'l1':     lambda X: self.l_norm(X, 1, norms),
            'l2':     lambda X: self.l_norm(X, 2, norms),
            'minmax': lambda X: self.min_max(X, minimum, maximum),
                }

        return methods[self.method](X)
    
    def zscore(self, normX, means=None, stds=None):
        '''
        Perform zscore normalization on a numeric pandas dataframe.

        Returns
        -------
        normalized X : pandas dataframe
            Z-score normalized pandas dataframe.
        '''

        self.means = means if means is not None else normX.mean(axis=0)
        self.stds = stds if stds is not None else normX.std(axis=0)

        return normX.subtract(self.means,axis=1) / self.stds

    def log(self, normX):
        '''
        Perform log normalization on a numeric pandas dataframe.

        Returns
        -------
        normalized X : pandas dataframe
            Log normalized pandas dataframe.
        '''
        return np.log(normX + 1e-10)
    
    def l_norm(self, normX, ord, norms=None):
        '''
        Perform L1 or L2 normalization on a numeric pandas dataframe.

        Parameters
        ----------
        ord : {1, 2}
            Order of normalization to be performed. 

        Returns
        -------
        normalized X : pandas dataframe
            L1 or L2 normalized pandas dataframe.

        Notes
        -----
        Normalizes rows (not columns) by their L1/L2 norms such the sum of their absolutes
        (L1) or their squares (L2) equal 1. As such, this provides insight into the relative 
        magnitude of sample's features relative to its other features rather than to the same
        feature across the population.
        '''
        if norms:
            self.norms = norms
        else:
            self.norms = np.linalg.norm(normX, ord=ord, axis=1)
            self.norms[self.norms == 0] = 1

        return normX.div(self.norms, axis=0)
    
    def min_max(self, normX, minimum, maximum):
        '''
        Perform min-max normalization on a numeric pandas dataframe.

        Returns
        -------
        normalized X : pandas dataframe
            Min-max normalized pandas dataframe.
        '''
        scale = maximum - minimum
        return normX.subtract(normX.min(axis=0), axis=1) / (normX.max(axis=0) - normX.min(axis=0)) * scale + minimum
    
class Splitter:
    def __init__(self, method='kfold', validationSplit=0.2, groups=5, bagging=True):
        self.method = method
        self.validationSplit = validationSplit
        self.groups = groups
        self.bagging = bagging

        if self.method not in ['lazy', 'kfold']:
            warnings.warn(f"Invalid or Nonetype method ({method}); defaulting to kfold.", UserWarning)
            self.method = 'kfold'
        
    def __call__(self, X, y):
        methods = {
            'lazy': lambda X, y: self.create_lazy_split(X, y),
            'kfold': lambda X, y: self.create_kfold_cross_validation_split(X, y)
        }

        methods[self.method](X,y)

    def create_set(self, data, idx):
        return data.iloc[idx]

    def create_lazy_split(self, X, y):
        validationIndices = np.random.choice(y.shape[0], size=int(y.shape[0] * self.validationSplit), replace=False)
        trainIndices = np.setdiff1d(np.arange(y.shape[0]), validationIndices)

        self.trainX, self.trainY = [self.create_set(X, trainIndices)], [self.create_set(y, trainIndices)]
        self.validationX, self.validationY = [self.create_set(X,validationIndices)], [self.create_set(y, validationIndices)]

    def create_kfold_cross_validation_split(self, X, y):
        if self.bagging:
            validationIndices = [np.random.choice(y.shape[0], size=int(y.shape[0] / self.groups), replace=True) 
                                    for i in range(self.groups)]
        else:
            indices = np.random.permutation(y.shape[0])
            validationIndices = [indices[int(np.round(y.shape[0] / self.groups*i)):int(np.round(y.shape[0] / self.groups*(i + 1)))] \
                                for i in range(self.groups)]
        
        trainIndices = [np.setdiff1d(np.arange(y.shape[0]), validationIndices[i]) for i in range(self.groups)]

        self.trainX = [self.create_set(X, trainIndices[i]) for i in range(self.groups)]
        self.trainY = [self.create_set(y, trainIndices[i]) for i in range(self.groups)]
        self.validationX = [self.create_set(X, validationIndices[i]) for i in range(self.groups)]
        self.validationY = [self.create_set(y, validationIndices[i]) for i in range(self.groups)]
    
class Method:
    def __init__(self, imputationMethod='lazy', normalizationMethod='zscore', splitMethod='kmeans',
                 metrics=('mse',), validationSplit=0.2, inlineDiagnostics=False, finalDiagnostics=True, **kwargs):
        onehotCols = kwargs.pop("onehotCols", None)
        directCols = kwargs.pop("directCols", None)
        ordinalCols = kwargs.pop("ordinalCols", None)
        dropCols = kwargs.pop("dropCols", None)
        splitGroups = kwargs.pop("splitGroups", 5)
        
        self.quantifier = self.Quantifier(onehotCols = onehotCols, directCols = directCols, ordinalCols = ordinalCols, dropCols=dropCols)
        self.imputer = self.Imputer(method = imputationMethod)
        self.normalizer = self.Normalizer(method = normalizationMethod)
        self.splitter = self.Splitter(method = splitMethod, groups=splitGroups, validationSplit=validationSplit)
        self.inlineDiagnostics, self.finalDiagnostics = inlineDiagnostics, finalDiagnostics
        self.diagnostics = Diagnostics(metrics = metrics)

    def transform(self, X, means=None, stds=None, maximum=None, minimum=None, norms=None):
        return self.normalizer(self.imputer(self.quantifier(X)), maximum=maximum, minimum=minimum, means=means, stds=stds,
                               norms=norms)

    def _get_splits(self, X, y):
        self.splitter(X=X, y=y)

    def train(self, X, y, modelMethod='logreg', lossMethod='binary_cross_entropy', optimizationMethod='gradient_descent', 
                 initialGuess='zeros', lr=0.05, lrMin=0.01, lrDecay=1e-4, lrUpdate='percent', maxEpochs=1e3, minEpochs=10, maxTreeDepth=3,
                 minGroupSize=5, nTrees=100, nCols=0.2, nRows=0.5, nBatch=64, momentumMethod=None, momentumFactor=0.9, 
                 momentumWarmup=1000, lossThreshold=0, lossWindow=100, gradThreshold=1e-5, splitMethod='histogram', maxBoosterDepth=10,
                 temperature=1e5, cooling='exponential', coolingRate=-1e-4, stepSize=0.1, sigmoid=False, binary=False):
        if modelMethod in ['decision_tree', 'random_forest']:
            self.modelPrototype = model.RandomForest(modelMethod=modelMethod, lossMethod=lossMethod, maxTreeDepth=maxTreeDepth, 
                                            minGroupSize=minGroupSize, nTrees=nTrees, nCols=nCols, nRows=nRows, splitMethod=splitMethod)
        elif modelMethod in ['gradient_boosting']:
            self.modelPrototype = model.GradientBooster(modelMethod=modelMethod, lossMethod=lossMethod, maxTreeDepth=maxTreeDepth, 
                                            minGroupSize=minGroupSize, nCols=nCols, nRows=nRows, splitMethod=splitMethod,
                                            maxBoosterDepth=maxBoosterDepth, lr=lr)
        elif optimizationMethod in ['gradient_descent', 'minibatch_gradient_descent', 'stochastic_gradient_descent']:
            self.modelPrototype = model.GradientDescent(modelMethod=modelMethod, optimizationMethod=optimizationMethod, lossMethod=lossMethod, 
                                               lr=lr, lrMin=lrMin, lrDecay=lrDecay, lrUpdate=lrUpdate, initialGuess=initialGuess, 
                                               maxEpochs=maxEpochs, minEpochs=minEpochs, gradThreshold=gradThreshold, 
                                               lossThreshold=lossThreshold, lossWindow=lossWindow, sigmoid=sigmoid, binary=binary, 
                                               nBatch=nBatch, momentumMethod=momentumMethod, momentumFactor=momentumFactor, 
                                               momentumWarmup=momentumWarmup)
        elif optimizationMethod in ['simulated_annealing']:
            self.modelPrototype = model.SimulatedAnnealing(modelMethod=modelMethod, optimizationMethod=optimizationMethod, lossMethod=lossMethod,
                                                  temperature=temperature, cooling=cooling, coolingRate=coolingRate,
                                                  stepSize = stepSize, initialGuess=initialGuess, maxEpochs=maxEpochs, minEpochs=minEpochs, 
                                                  gradThreshold=gradThreshold, lossThreshold=lossThreshold, 
                                                  lossWindow=lossWindow, sigmoid=sigmoid, binary=binary)

        self._get_splits(X=X, y=y)
        self.y_hats = [None for _ in range(len(self.splitter.trainX))]
        self.diagnosticDicts = []
        self.models = np.empty((len(self.splitter.trainX),), dtype=object)

        for i in range(len(self.splitter.trainX)):
            startTime = time.time()
            self.models[i] = copy.deepcopy(self.modelPrototype)
            self.models[i](self.splitter.trainX[i], self.splitter.trainY[i])
            self.models[i].runtime = np.round(time.time() - startTime, 2)
            self.y_hats[i] = self.models[i]._calc_y_hat(X=self.splitter.validationX[i], theta=self.models[i].thetas[-1], binary=True)
            self.diagnostics(y=self.splitter.validationY[i].copy(), y_hat=np.array(self.y_hats[i]).copy(), output=self.inlineDiagnostics,
                             runtime=self.models[i].runtime)
            self.diagnosticDicts.append(self.diagnostics.diagnostics.copy())

        if self.finalDiagnostics:
            self.ensembleDict, self.ensembleStr = self.diagnostics.calc_ensemble_diagnostics(self.diagnosticDicts)
            print(self.ensembleStr)

    def predict(self, X, binary=False):
        yPred = np.empty((len(self.models), X.shape[0]))
        for i in range(len(self.models)):
            yPred[i] = self.models[i]._calc_y_hat(X=X, theta=self.models[i].theta, binary=binary)

        if binary:
            yPred = np.round(np.mean(yPred, axis=0))
        else:
            yPred = np.mean(yPred, axis=0)
        
        return yPred

    

    
        
    class DimensionalityReducer:
        '''
        Class to reduce the dimensionality of a feature space.

        Attributes
        ----------
        nDims : int
            Number of dimensions to reduce the feature space to
        method : {'PCA'}
            Method for reducing the feature space
        '''

        def __init__(self, nDims, method='PCA'):
            self.nDims = nDims

            if method not in ['PCA']:
                warnings.warn(f"Invalid or Nonetype method ({method}); defaulting to PCA.", UserWarning)
                self.method = 'PCA'
            else:
                self.method = method

        def __call__(self, X):
            methods = {
                'PCA': lambda X: self.PCA(X),
                    }
            
            return methods[self.method](X)
            

        def PCA(self, X):
            pass