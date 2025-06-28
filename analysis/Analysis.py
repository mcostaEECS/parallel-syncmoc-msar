from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import cv2
import seaborn as sns
import os
from load_data import load_data
import pyprind
import re
import scipy.io as sio
import datetime
import rasterio
import numpy as np
from matplotlib.patches import Rectangle
import rasterio.plot
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.plot import show_hist
from osgeo import gdal
import numpy as np
import time
from numpy import zeros,sqrt, mean,linspace,concatenate, cumsum
from scipy.stats import norm
import cv2
from matplotlib import pyplot as plt
import scipy.io as sio
from PIL import Image
from skimage import data
from skimage.morphology import disk
from skimage.filters.rank import median
import cv2
from load_data import load_data
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from matplotlib.widgets  import RectangleSelector
import matplotlib.patches as patches
import scienceplots
from matplotlib.collections import LineCollection



# unification of plots 250 and 500 under LT (CT - CC) with pdfs and thrsholds


#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
    
def moving_average(im, K):
    kernel1 = np.ones((K,K),np.float32)/(K**2)
    Filtered_data = cv2.filter2D(src=im, ddepth=-1, kernel=kernel1) 
    return Filtered_data



def test(Itest, Iref):
    
 
     
    TestVec =Itest.ravel()
    RefVec = Iref.ravel()
    
    
    rho = np.corrcoef(TestVec,RefVec)[0][1]
    unrho = np.sqrt(1-rho**2)
 
    th = 1.2
    CD0 = []
    for i in range(len(TestVec)):
        #resTarget = TestVec[i]*rho + RefVec[i]*unrho # degree of persistence
        
        if TestVec[i] > th*RefVec[i]:
            resTarget = TestVec[i]*rho + RefVec[i]*unrho # degree of persistence
            
        else:
            resTarget = TestVec[i]
            
            
    
        CD0.append(resTarget)
        
    return CD0

def MCnAR1LT(start1,start2, end1, end2, N,K,par, scene):
    
    ItestCT = par[25][start1:end1,start2:end2]  
        
    IrefA = par[0][start1:end1,start2:end2]
    IrefB = par[1][start1:end1,start2:end2]
    IrefC = par[2][start1:end1,start2:end2]
    IrefD = par[3][start1:end1,start2:end2]
    IrefE = par[4][start1:end1,start2:end2]
    IrefF = par[5][start1:end1,start2:end2]
    
    IrefG = par[6][start1:end1,start2:end2]
    IrefH = par[7][start1:end1,start2:end2]
    IrefI = par[8][start1:end1,start2:end2]
    IrefJ = par[9][start1:end1,start2:end2]
    IrefK = par[10][start1:end1,start2:end2]
    IrefL = par[11][start1:end1,start2:end2]
    
    IrefM = par[12][start1:end1,start2:end2]
    IrefN = par[13][start1:end1,start2:end2]
    IrefO = par[14][start1:end1,start2:end2]
    IrefP = par[15][start1:end1,start2:end2]
    IrefQ = par[16][start1:end1,start2:end2]
    IrefR = par[17][start1:end1,start2:end2]
    
    
    ar1 = test(ItestCT, IrefA);ar2 = test(ItestCT, IrefB);ar3 = test(ItestCT, IrefC)
    ar4 = test(ItestCT, IrefD);ar5 = test(ItestCT, IrefE);ar6 = test(ItestCT, IrefF)
    
    ar7 = test(ItestCT, IrefG);ar8 = test(ItestCT, IrefH);ar9 = test(ItestCT, IrefI)
    ar10 = test(ItestCT, IrefJ);ar11 = test(ItestCT, IrefK);ar12 = test(ItestCT, IrefL) 
    
    ar13 = test(ItestCT, IrefM);ar14 = test(ItestCT, IrefN);ar15 = test(ItestCT, IrefO)
    ar16 = test(ItestCT, IrefP);ar17 = test(ItestCT, IrefQ);ar18 = test(ItestCT, IrefR) 

   
    
    p1 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar1).flatten())[0][1])
    p2 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar2).flatten())[0][1])
    p3 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar3).flatten())[0][1])
    p4 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar4).flatten())[0][1])
    p5 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar5).flatten())[0][1])
    p6 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar6).flatten())[0][1])
    
    p7 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar7).flatten())[0][1])
    p8 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar8).flatten())[0][1])
    p9 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar9).flatten())[0][1])
    p10 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar10).flatten())[0][1])
    p11 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar11).flatten())[0][1])
    p12 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar12).flatten())[0][1])
    
    p13 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar13).flatten())[0][1])
    p14 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar14).flatten())[0][1])
    p15 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar15).flatten())[0][1])
    p16 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar16).flatten())[0][1])
    p17 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar17).flatten())[0][1])
    p18 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar18).flatten())[0][1])
    

    
    pLT = [p1,p2,p3,p4, p5, p6, p7, p8, p9,p10,p11,p12, p13, p14, p15, p16, p17, p18]                      
    arLT = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8, ar9, ar10, ar11, ar12, ar13, ar14, ar15, ar16, ar17, ar18]
      
   
    IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:2]  # sorting
    print(IdxLT)
    IdxST = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:2]  # sorting
    criteria = ['avg', 'min']
    criteria = criteria[0]
    
    TestVec =ItestCT.ravel()
    
    
    CDmcAvg = []; CDmcMin=[]
    for i in range(len(TestVec)):
        if criteria == 'avg':
            ST1 = arLT[IdxST[0]][i]
            LT1 = arLT[IdxLT[0]][i]
            
            res = TestVec[i] - LT1  # change map from clutter reduction
            
        elif criteria == 'min':
            res =arLT[IdxLT[0]][i]  # change map

        CDmcAvg.append(res)

    ICD3 = np.reshape(CDmcAvg, (N,N))
    resARAvg3=moving_average(ICD3, K)
    
    
        
        
        
    
    return resARAvg3.ravel()

def MCnAR1ST(start1,start2, end1, end2, N,K,par, scene):
    
    ItestCT = par[25][start1:end1,start2:end2]  #36 / 18 limite de memoria
        
    IrefA = par[0][start1:end1,start2:end2]
    IrefB = par[1][start1:end1,start2:end2]
    IrefC = par[2][start1:end1,start2:end2]
    IrefD = par[3][start1:end1,start2:end2]
    IrefE = par[4][start1:end1,start2:end2]
    IrefF = par[5][start1:end1,start2:end2]
    
    
    
    ar1 = test(ItestCT, IrefF);ar2 = test(ItestCT, IrefE);ar3 = test(ItestCT, IrefE)
    ar4 = test(ItestCT, IrefD);ar5 = test(ItestCT, IrefE);ar6 = test(ItestCT, IrefF)
    
   
    
    p1 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar1).flatten())[0][1])
    p2 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar2).flatten())[0][1])
    p3 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar3).flatten())[0][1])
    p4 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar4).flatten())[0][1])
    p5 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar5).flatten())[0][1])
    p6 = abs(np.corrcoef(np.array(ItestCT).flatten(),np.array(ar6).flatten())[0][1])
    
   
    

    
    pLT = [p1,p2,p3,p4, p5, p6]                      
    arLT = [ar1, ar2, ar3, ar4, ar5, ar6]
      
   
    IdxLT = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:2]  # sorting
    print(IdxLT)
    IdxST = sorted(range(len(pLT)), key=lambda sub: pLT[sub], reverse=True)[:2]  # sorting
    criteria = ['avg', 'min']
    criteria = criteria[0]
    
    TestVec =ItestCT.ravel()
    
    
    CDmcAvg = []; CDmcMin=[]
    for i in range(len(TestVec)):
        if criteria == 'avg':
            ST1 = arLT[IdxST[0]][i]
            LT1 = arLT[IdxLT[0]][i]
            
            res = TestVec[i] - LT1  # change map from clutter reduction
            
        elif criteria == 'min':
            res =arLT[IdxLT[0]][i]  # change map

        CDmcAvg.append(res)

    ICD3 = np.reshape(CDmcAvg, (N,N))
    resARAvg3=moving_average(ICD3, K)
    
    
        
        
        
    
    return resARAvg3.ravel()


if __name__ == "__main__":

    T = 1  # 0 = glrt / 1 = lmp
    TestType = ['ARn1', 'LMP']
    block = [250,500]
    test_type = ['ST', 'MT', 'LT']
    block= block[1]
    test_type = test_type[2]
    


    s = 6
    #s = s + 3
    par=load_data(TestType[0])[s]
    tp=par[23]; tp =  np.fliplr(tp); TP =par[24]
    
   
  
    if block == 250:
        N = 250; K=9
        
        if test_type == 'LT':
            
            scene = 'CC'   
            start1=375; end1 = 625; start2=675; end2 = 925 # K1 250 C
            resAvgCC = MCnAR1LT(start1,start2, end1, end2, N,K, par, scene)
            
            scene = 'CT'
            start1=375; end1 = 625; start2=475; end2 = 725 # K1 250 T        
            resAvgCT = MCnAR1LT(start1,start2, end1, end2, N,K, par,scene)
            
            
            percentiles= np.array([75])
            x_p = np.percentile(resAvgCT, percentiles)
            y_p = percentiles/100.0
            
            quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
            iqr = quartile_3 - quartile_1
            
            
            upper_bound = quartile_3 + (iqr * 1.5)
            
        elif test_type == 'ST':
                
            
            scene = 'CC'        
            start1=375; end1 = 625; start2=675; end2 = 925 # K1 250 C
            resAvgCC = MCnAR1ST(start1,start2, end1, end2, N,K, par, scene)
            scene = 'CT'        
            start1=375; end1 = 625; start2=475; end2 = 725 # K1 250 T    
            resAvgCT = MCnAR1ST(start1,start2, end1, end2, N,K, par,scene)
            # resAvgMT = MCnAR1MT(start1,start2, end1, end2, N,K, par)
            # resAvgST = MCnAR1ST(start1,start2, end1, end2, N,K, par)
            
        
    
    
    
            percentiles= np.array([75])
            x_p = np.percentile(resAvgCT, percentiles)
            y_p = percentiles/100.0
            
            quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
            iqr = quartile_3 - quartile_1
            
            
            upper_bound = quartile_3 + (iqr * 1.5)
            
    elif block == 500:
        N = 500; K=9
        
        
    
        
        if test_type == 'LT':
            
            scene = 'CC'   
            start1=250; end1 = 750; start2=750; end2 = 1250 # K1 500 C
            resAvgCC = MCnAR1LT(start1,start2, end1, end2, N,K, par, scene)
            
            scene = 'CT'
            start1=250; end1 = 750; start2=350; end2 = 850 # K1 500 T       
            resAvgCT = MCnAR1LT(start1,start2, end1, end2, N,K, par,scene)
            
            
            percentiles= np.array([75])
            x_p = np.percentile(resAvgCT, percentiles)
            y_p = percentiles/100.0
            
            quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
            iqr = quartile_3 - quartile_1
            
            
            upper_bound = quartile_3 + (iqr * 1.5)
            
        elif test_type == 'ST':
                
            
            scene = 'CC'        
            start1=250; end1 = 750; start2=750; end2 = 1250 # K1 500 C
            resAvgCC = MCnAR1ST(start1,start2, end1, end2, N,K, par, scene)
            scene = 'CT'        
            start1=250; end1 = 750; start2=350; end2 = 850 # K1 500 T      
            resAvgCT = MCnAR1ST(start1,start2, end1, end2, N,K, par,scene)
            # resAvgMT = MCnAR1MT(start1,start2, end1, end2, N,K, par)
            # resAvgST = MCnAR1ST(start1,start2, end1, end2, N,K, par)
            
        
    
    
    
            percentiles= np.array([75])
            x_p = np.percentile(resAvgCT, percentiles)
            y_p = percentiles/100.0
            
            quartile_1, quartile_3 = np.percentile(resAvgCT, [25, 75])
            iqr = quartile_3 - quartile_1
            
            
            upper_bound = quartile_3 + (iqr * 1.5)
    
          
          


        

 # # #------------------------ Graphic Analysis -------------------------#
 
    DPI = 250
    plt.rc('text', usetex=True)
    DPI =250


    SMALL_SIZE = 10
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 12

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    

    
 
    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2), dpi=DPI) #, constrained_layout=True)
            ax1.set_prop_cycle(color=['rosybrown', 'slategray', 'red', 'gray']) #linewidth=2
            # ax1.plot(CD1, color='rosybrown', linewidth=2, label=r'target-clutter') 
            # ax1.plot(CD0, color='slategray', linewidth=2,  label=r'clutter-clutter') 
            #plt.gca().invert_yaxis()
            
            #sns.distplo(cd0, hist = True, kde = True, kde_kws = {'linewidth': 2})
            # sns.kdeplot(CD0, linewidth=1, ax=ax1, label=r'clutter-clutter') 
            # sns.kdeplot(CD1, linewidth=1, ax=ax1, label=r'target-clutter') 
            
            # data = np.random.randn(200)
            # print(data[0:100])
            # print(np.array(CD0)[0:100])
            
            #ax1.plot(resARAvgVecC3, color='', linewidth=2,  label=r'target-clutter (Avg)') 
            ax1.plot(resAvgCT, color='rosybrown', linewidth=2,  alpha=0.25, label=r'clutter-target') 
            #ax1.plot(resAvgMT, color='slategray', linewidth=2,  alpha=0.25,label=r'target-clutter')
            ax1.plot(resAvgCC, color='gray', linewidth=2,alpha=0.35, label=r'clutter-clutter')
            
            
   
            # ax1.distplot(cd0, 30,  hist = False, kde = True, kde_kws = {'linewidth': 3}, label=r'clutter-clutter') 
            # ax1.distplot(cd1, 30,  hist = False, kde = True, kde_kws = {'linewidth': 3}, label=r'target-clutter') 
            
            
            #hist = False, kde = True, kde_kws = {'linewidth': 3}
            # show_hist(cd0, bins=200, histtype='step',
            # lw=1, alpha=0.8, ax=ax1)
            # show_hist(cd1, bins=200, histtype='step',
            # lw=1, alpha=0.8,  ax=ax1)
            
            #ax1.semilogx(df.flops, df.data_pct, marker = '*', linewidth=2) # test append(s)
  
            #ax1.set(**pparam2)
            ax1.legend(prop={'size': 14})
           
            #ax1.invert_xaxis()
            ax1.set_ylabel(r'Magnitue', fontsize=10)  
            ax1.set_xlabel(r'pixels [$1\times N$]', fontsize=10)  
            ax1.xaxis.set_label_coords(.5, -.2)
            ax1.yaxis.set_label_coords(-0.14, 0.5)

            
            meanColor = ['k', 'navy'] 
            ax2.set_prop_cycle(color=['slategray', 'rosybrown', 'red', 'gray']) #linewidth=2
            
            #ax2.loglog(freqs1, psd1, 'rosybrown', freqs0, psd0, 'slategray', alpha=0.7)
            #ax2.loglog(freqs1, psd1, 'rosybrown', freqs0, psd0, 'slategray', alpha=0.7)
            # sns.kdeplot(np.array(CD0), color='slategray',linestyle='-.', ax=ax2) 
            # sns.kdeplot(np.array(CD1),color='rosybrown', linestyle='-.', ax=ax2)
            
            sns.kdeplot(np.array(resAvgCT), color='rosybrown', alpha=0.25,ax=ax2) 
            plt.axvline(upper_bound, 0,1, color='red', alpha=0.35) 
            #sns.kdeplot(np.array(resAvgMT),color='slategray', alpha=0.25,ax=ax2) 
            sns.kdeplot(np.array(resAvgCC),color='gray', alpha=0.35,ax=ax2) 
            
            #sns.kdeplot(np.array(resARAvgVecT3), color='slategray', ax=ax2) 
            #sns.kdeplot(np.array(cd1Avg.ravel()),color='rosybrown', ax=ax2) 
            
            
           # ax2.set(**pparam2)
            ax2.legend(prop={'size': 5})
            ax2.set_ylabel(r'Density', fontsize=10)  
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r'Magnitude', fontsize=10)  
            ax2.xaxis.set_label_coords(.5, -.2)
            ax2.yaxis.set_label_coords(-0.14, 0.5)
            ax2.set_xlim([0, 0.5])
            ax2.yaxis.set_tick_params(labelleft=False, labelright=True)


            ax1.legend(loc='lower center', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=4)
            


            # ax1.text(0.5,-0.09, "(a)", size=12, ha="center", 
            #         transform=ax1.transAxes)
            # ax2.text(0.5,-0.09, "(b)", size=12, ha="center", 
            #         transform=ax2.transAxes)
    path='/home/marcello-costa/workspace/SPL/Plots/'
    namefile = 'A3_%s_N_%d.png'%(test_type,block)
 
    fig.savefig(path + namefile, dpi=DPI)
    fig.tight_layout()
    fig.tight_layout()
            
          

    plt.show()      

            
        
    
 












