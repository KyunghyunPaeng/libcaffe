import os
import numpy as np
import matplotlib.pyplot as plt

class Monitor(object):
    def __init__(self, log_file):
        self._log_file = log_file
        if not os.path.exists(self._log_file):
            assert False, 'Log file does not exist'
    
    def monitoring(self, show=False):
        trn_logs, val_logs, trn_done = self._parse_log()
        trn_iter, trn_loss = zip(*trn_logs)
        val_iter, val_accu, val_loss = zip(*val_logs)
        optimal_model_loss = val_iter[np.argmin(val_loss)]
        optimal_model_accu = val_iter[np.argmax(val_accu)]
        total_res = [(trn_iter,trn_loss),(val_iter,val_loss),(val_iter,val_accu)]
        if show:
            sub_trn_iter = []
            sub_trn_loss = []
            for i in val_iter:
                sub_trn_iter.append(i)
                sub_trn_loss.append(trn_loss[trn_iter.index(i)])
            plt.figure()
            plt.plot(sub_trn_iter, sub_trn_loss, '-or', label='Training loss')
            plt.plot(val_iter, val_accu, '-+g', label='Validation accuracy')
            plt.plot(val_iter, val_loss, '-*b', label='Validation loss')
            plt.ylim(0,1)
            plt.xlabel('Iteration')
            plt.legend(loc='best')
            plt.show()
            total_res = [(sub_trn_iter,sub_trn_loss),(val_iter,val_loss),(val_iter,val_accu)]
        return {'value':total_res, 'trn_done':trn_done, 'loss':optimal_model_loss, 'accu':optimal_model_accu}
    
    def _parse_log(self):
        with open(self._log_file, 'r') as f:
            logs = f.readlines()
        trn_lines = []
        val_lines = []
        trn_done = False
        for lineidx in xrange(len(logs)):
            line = logs[lineidx]
            if 'Iteration' in line:
                if 'loss' in line:
                    splitted = line.split()
                    trn_iter_num = int(splitted[5][:-1])
                    trn_loss = float(splitted[-1])
                    trn_lines.append((trn_iter_num,trn_loss))
                elif 'Testing net' in line:
                    try:
                        splitted = line.split()
                        val_iter_num = int(splitted[5][:-1])
                        
                        splitted_accu = logs[lineidx+1].split()
                        val_accu = float(splitted_accu[-1])
 
                        splitted_loss = logs[lineidx+2].split()
                        val_loss = float(splitted_loss[-2])
 
                        val_lines.append((val_iter_num, val_accu, val_loss))
                    except:
                        continue
            if 'Optimization Done.' in line:
                trn_done = True
        return trn_lines, val_lines, trn_done
