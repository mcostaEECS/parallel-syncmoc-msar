from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import pyprind
import time
import psutil
import datetime
import pyprind
import time
import psutil
import re
import os
import signal
import subprocess
from datetime import datetime




def SPLSim(test_type, N, K, lags):
    
    


    
    campaignRuntime = 'Runtime_%s_N_%d_K_%d'%(test_type, N, K)
    campaignPower = '%s_N_%d_K_%d'%(test_type, N, K)
    path = '/home/marcello-costa/workspace/SPL/Out/logs/'
    
    nroPairs = 4
    campaign = ['K1', 'K1', 'K1','K1'] # emulate a 1k x 1k image
    
    cmdBAT = 'python3 PowerMeasure.py %s' %campaignPower
    proBAT = subprocess.Popen(cmdBAT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    time.sleep(20)
    
    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaignPower)
    for s in range(nroPairs):
        
        #s = s + 1
 
        
        
        name_file = path+campaignRuntime+'.txt'
        with open(name_file, 'a') as f:
            pathTest = '/home/marcello-costa/workspace/SPL/bin/'
         
            os.chdir(pathTest)
            if test_type == 'AR1MoCseq':
               par = 1
               com = './%s_%sAR1seq %s_%sAR1seq.1.txt +RTS -N%s' % (campaign[s], lags, campaign[s], lags, par)
                
            elif test_type == 'AR1MoCpar':
                par=4
                com = './%s_%sAR1par %s_%sAR1par.1.txt +RTS -N%s' % (campaign[s], lags, campaign[s], lags, par)
           
            
            START = time.time()
            os.system(com)
            END = time.time()
            dt = datetime.now()
            dt.microsecond   
            
            timeAR1 = dt
            Nt = 0; timeCFAR = 0

            f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'HT_runtime_'+str(timeAR1)+';'+'cfar_runtime_'+str(timeCFAR)+';'+'Threshold_op_'+str(Nt)+'\n')
            bar.update()
    
    #os.killpg(os.getpgid(proBAT.pid), signal.SIGTERM)
                
            

if __name__ == "__main__":
    
    idxTest = 1
    
    test_type = ['AR1MoCseq', 'AR1MoCpar']
    lags = [1,3]  # 1 for 'AR1MoC'/ and 6 for 'AR6MoC'
    N = [500,250]
    
    K = 9
   
    
    
    #SPLSim(test_type[0], N[0], K, lags[1]) # seq 
    SPLSim(test_type[1], N[1], K, lags[1]) # par
        
    
    
        
        
        
        