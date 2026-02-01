import numpy as np
import pandas as pd
import warnings

class Diagnostics:
    def __init__(self, metrics=('accuracy', 'mse'), **diagnosticParams):
        self.diagnostics = {}
        METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "mse")

        for m in METRICS:
            setattr(self, m, m in metrics)

    def __call__(self, y, y_hat, output=True, runtime=None):
        self.diagnostics['Runtime'] = runtime
        self.diagnostics['Accuracy'] = self.calc_accuracy(y, y_hat) if self.accuracy else None
        self.diagnostics['Precision'] = self.calc_precision(y, y_hat) if self.precision else None
        self.diagnostics['Recall'] = self.calc_recall(y, y_hat) if self.recall else None
        self.diagnostics['Specificity'] = self.calc_specificity(y, y_hat) if self.specificity else None
        self.diagnostics['F1'] = self.calc_f1_score(y, y_hat) if self.f1 else None
        self.diagnostics['MSE'] = self.calc_mse(y, y_hat) if self.mse else None

        if output:
            diagnosticStr = ''

            for key, val in self.diagnostics.items():
                diagnosticStr += f'{key}: {val}, ' if val else ''

            print(diagnosticStr[:-2])

    def calc_ensemble_diagnostics(self, diagnosticDicts):
        self.ensembleDict = {}
        self.ensembleStr = ''

        for key in diagnosticDicts[0].keys():
            if diagnosticDicts[0][key]:
                self.ensembleDict[f'Average {key}'] = np.round(np.mean([diagnosticDicts[i][key] for i in range(len(diagnosticDicts))]), 3)
                self.ensembleDict[f'{key} Variance'] = np.round(np.var([diagnosticDicts[i][key] for i in range(len(diagnosticDicts))]), 5)

                self.ensembleStr += f'Average {key}: {np.round(np.mean([diagnosticDicts[i][key] for i in range(len(diagnosticDicts))]), 3)}, '
                self.ensembleStr += f'{key} Variance: {np.round(np.var([diagnosticDicts[i][key] for i in range(len(diagnosticDicts))]), 5)}, '

        self.ensembleStr = self.ensembleStr[:-2]
        
        return self.ensembleDict, self.ensembleStr

    def calc_accuracy(self, y, y_hat):
        return np.round(100 - np.mean(np.abs(y - y_hat))*100, 3)
    
    def calc_mse(self, y, y_hat):
        return np.round(np.mean((y - y_hat) ** 2), 3)

    def calc_precision(self, y, y_hat):
        return np.round(np.sum((y_hat == y)[y_hat == 1] / np.sum(y_hat == 1))*100, 3)
    
    def calc_recall(self, y, y_hat):
        return np.round(np.sum((y_hat == y)[y_hat == 1] / np.sum(y == 1))*100, 3)
    
    def calc_specificity(self, y, y_hat):
        return np.round(np.sum((y_hat == y)[y_hat == 0] / np.sum(y_hat == 0))*100, 3)
    
    def calc_f1_score(self, y, y_hat):
        return 2 * self.calc_precision(y, y_hat) * self.calc_recall(y, y_hat) / (self.calc_precision(y, y_hat) + self.calc_recall(y, y_hat))
    