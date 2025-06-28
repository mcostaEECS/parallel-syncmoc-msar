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
from Classifier import Classifier
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



#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
    
def moving_average(im):
    kernel1 = np.ones((7,7),np.float32)/49
    Filtered_data = cv2.filter2D(src=im, ddepth=-1, kernel=kernel1) 
    return Filtered_data



def Classifier(ICD,tp,pfa, TestType):

    radius = 12

    Imc=np.array(ICD)

    if TestType == 'GLRT':

        ImAMax = Imc.max(axis=0).max(axis=0)
        th=pfa
        (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)
        im_bwN = cv2.normalize(im_bw, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    elif TestType == 'LMP':
        ImAMax = Imc.max(axis=0).max(axis=0)
        th=pfa
        (thresh, im_bw) = cv2.threshold(Imc, th, ImAMax, cv2.THRESH_BINARY)

        im_bwN = im_bw


    #OPERACOES MORFOLOGICAS:
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    erosion = cv2.erode(im_bwN,kernel,iterations = 1)
    dilation1 = cv2.dilate(erosion,kernel,iterations = 1)
    dilation2 = cv2.dilate(dilation1,kernel,iterations = 1)
    Imc=dilation2

    fig = plt.figure('test')
    plt.suptitle('Binarizada')
    
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(Imc, cmap = plt.cm.gray)
    plt.axis("off")
    
    plt.show() 
    
    results= cv2.connectedComponentsWithStats(np.array(Imc, dtype=np.uint8),8)

    num_Objects= results[0]
    centroids = results[3]
    centro = centroids[np.argsort(centroids[:, 0])]

    
    if num_Objects > 0:

        outlines = []
        detected = [] 
        for k in range(len(tp)):
            for m in range(len(centro)):
                if (np.abs(centro[m][0] - tp[k][0]) < radius) and (np.abs(centro[m][1] - tp[k][1]) < radius):
                    detected.append(m)
                else:
                    outlines.append(m)
        
        detected_dict = {i:detected.count(i) for i in detected}
        outlines_dict = {i:outlines.count(i) for i in outlines}
        detected_idx = list(detected_dict.keys())
        outlines_idx = list(outlines_dict.keys())

        outlines_idx = [ x for x in outlines_idx if not x in detected_idx]

        potential_targets = num_Objects 
        detected_targets= len(detected_idx) # ok
        if detected_targets > len(tp):
            detected_targets = len(tp)
            falseAlarm = np.abs(potential_targets - len(tp))
        else:
            falseAlarm = np.abs(potential_targets - detected_targets)
    else:
        detected_targets = 0
        falseAlarm = 0
        
    mission_targets = len(tp)    

 
    f, ax = plt.subplots()
    
    for k in outlines_idx:
        #print(k)
        circleOutlines = plt.Circle((centro[k][0], centro[k][1]), 15, color='r', fill=False)
        ax.add_patch(circleOutlines)
        ax.imshow(Imc, cmap='gray', interpolation='nearest')
    for j in detected_idx:
        circleDetected = plt.Circle((centro[j][0], centro[j][1]), 15, color='g', fill=False)
        ax.imshow(Imc, cmap='gray', interpolation='nearest')
        ax.add_patch(circleDetected)

    plt.show()
    
    
    return detected_targets, falseAlarm, mission_targets


def spectrum1(h, dt=1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = len(h)
    npositive = nt//2
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice] 
    ft = np.fft.fft(h)[pslice]
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    # Normalization for Power Spectrum
    psraw /= nt**2
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    return freqs, psraw, psdraw


if __name__ == "__main__":

    T = 1  # 0 = glrt / 1 = lmp
    TestType = ['GLRT', 'LMP']


    # par=load_data(TestType[0])[3]
    # Itest=par[20][0:3000,0:2000] #Itest=par[0][0:3000,0:2000]
    # Iref=par[0][0:3000,0:2000]
    # tp=par[2]
    # #print()
    # TP =  np.fliplr(tp)
    
    s = 1
    #s = s + 3
    par=load_data(TestType[0])[s]
    tp=par[18]; tp =  np.fliplr(tp); TP =par[19]
    pair = s  
    Itest=par[20][0:3000,0:2000] #Itest=par[0][0:3000,0:2000]
    Iref=par[0][0:3000,0:2000]
    
    
    #    #### Target part 500*500
    # if pair==0 or pair==1 or pair==2 or pair==3 or pair==4 or pair==5:
    #     start1=450; end1 = 950; start2=350; end2 = 850 # S1
    # elif pair==6 or pair==7 or pair==8 or pair==9 or pair==10 or pair==11:
    #     start1=250; end1 = 750; start2=350; end2 = 850 # K1 
    # elif pair==12 or pair==13 or pair==14 or pair==15 or pair==16 or pair==17:
    #     start1=2050; end1 = 2550; start2=1100; end2 = 1600  # F1
    # elif pair==18 or pair==19 or pair==20 or pair==21 or pair==22 or pair==23:
    #     start1=2350; end1 = 2850; start2=950; end2 = 1450  # AF1  
    
   
    #### Target part 250*250
    if pair==0 or pair==1 or pair==2 or pair==3 or pair==4 or pair==5:
        start1=580; end1 = 810; start2=470; end2 = 700 # S1
    elif pair==6 or pair==7 or pair==8 or pair==9 or pair==10 or pair==11:
        start1=250; end1 = 750; start2=350; end2 = 850 # K1 
    elif pair==12 or pair==13 or pair==14 or pair==15 or pair==16 or pair==17:
        start1=2050; end1 = 2550; start2=1100; end2 = 1600  # F1
    elif pair==18 or pair==19 or pair==20 or pair==21 or pair==22 or pair==23:
        start1=2350; end1 = 2850; start2=950; end2 = 1450  # AF1  
    
        
    ItestTarget = par[20][start1:end1,start2:end2]  #36 / 18 limite de memoria
    IrefTarget = par[0][start1:end1,start2:end2]
    
    start1=1300; end1 = 1530; start2=1500; end2 = 1730  # F1
    
    IrefNonTarget=par[0][start1:end1,start2:end2]
    ItestNonTarget=par[20][start1:end1,start2:end2]
    
    
    
    
    
    ItestTarget = subarrays(ItestTarget, 115, 115) #2
    IrefTarget = subarrays(IrefTarget, 115, 115)
    IrefNonTarget = subarrays(IrefNonTarget, 115, 115)
    ItestNonTarget = subarrays(ItestNonTarget, 115, 115)
    
    ItestTarget = ItestTarget[2] #2
    IrefTarget = IrefTarget[2]
    IrefNonTarget = IrefNonTarget[2]
    ItestNonTarget = ItestNonTarget[2]
    
    # ItestTarget=moving_average(ItestTarget)
    # IrefTarget =moving_average(IrefTarget)
    
    # IrefNonTarget=moving_average(IrefNonTarget)
    # ItestNonTarget =moving_average(ItestNonTarget)
    
    
    
    
    
    

    # # path = '/home/marcelloee/Documentos/dLump/TestGLRT/'
    # # files_icd = [i for i in os.listdir(path) if i.endswith('.mat')]

    # # print(files_icd)
    # # ImageReadCD=sio.loadmat(path+files_icd[0])  #1 = GLRT
    # # CD=ImageReadCD['ICD']

    # ImageReadCD=sio.loadmat(path+files_icd[T])  #1 = LMP
    # CD2=ImageReadCD['ICD']


    # red noise estimation (TARGET vs NON Target)

    # IrefTarget=Iref[580:810,470:700]
    # ItestTarget=Itest[580:810,470:700]


    # # # Gaussian noise estimation (NON Target vs NON Target)

    # # IrefNonTarget=Iref[1300:1530,1500:1730]
    # # ItestNonTarget=Itest[1300:1530,1500:1730]


    #   # ########### plots ##############
    # #setup the figure
    # fig = plt.figure('test')
    # plt.suptitle('REF / TEST / Change')
    
    # ax = fig.add_subplot(1, 3, 1)
    # plt.imshow(IrefTarget, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # ax = fig.add_subplot(1, 3, 2)
    # plt.imshow(ItestTarget, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # ax = fig.add_subplot(1, 3, 3)
    # plt.imshow(ItestNonTarget, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # plt.show()  
    
    # print(g)

    # new image


    #Opening Image and resizing to 10X10 for easy viewing
    # image_test = np.array(ItestTarget.resize((230,230)))  #note: I used a local image
    # #print image
    # print (image_test)

    # #manipulate the array
    # x=np.array(image_test)
    # #convert to 1D vector
    # y=np.concatenate(x)
    # print (ItestTarget)



    

    #CD1 = ItestTarget.ravel()[4000:8000]-IrefTarget.ravel()[4000:8000]
    
    TestTargetVec = ItestTarget.ravel()
    RefTargetVec = IrefTarget.ravel()
    
    TestClutterVec = ItestNonTarget.ravel()
    RefClutterVec=IrefNonTarget.ravel()
    
    
    rhoTarget = np.corrcoef(TestTargetVec,RefTargetVec)[0][1]
    unrhoTarget = np.sqrt(1-rhoTarget**2)
    
    rhoClutter = np.corrcoef(TestClutterVec,RefClutterVec)[0][1]
    unrhoClutter = np.sqrt(1-rhoClutter**2)
    
    
    
    


    
    CD1=[]; CD0=[]
    for i in range(len(TestTargetVec)):
        resTarget = TestTargetVec[i]*rhoTarget + RefTargetVec[i]*unrhoTarget 
        resClutter = TestClutterVec[i]*rhoClutter + RefClutterVec[i]*unrhoClutter 
        
       
        CD1.append(resTarget); CD0.append(resClutter)
        
    # CD1n=[]
    # for i in range(len(CD1)):
    #     if CD1[i] > 0.1:
    #         res = CD1[i]
    #     else:
    #         res = 0
    #     CD1n.append(res)


    

    


    #CD0 =TestTargetVec - RefTargetVec
    # CD0 = TestClutterVec- RefClutterVec
    #CD = abs(CD)
    fs = 1000
    freqs1, ps1, psd1 = spectrum1(CD1, dt=1/fs)
    freqs0, ps0, psd0 = spectrum1(CD0, dt=1/fs)

    # fig, ax2 = plt.subplots(figsize=(12,4), dpi=250)
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(7,7), dpi=350)
    # #ax2.plot(Itest, linewidth=2, color='slategray',  alpha =0.2)
    
    # ax1.plot(CD0, linewidth=2, color='slategray') 
    # ax1.plot(CD1, linewidth=2, color='rosybrown', alpha=0.7) 
    # #ax2.plot(ItestTarget.ravel(), linewidth=2, color='indianred', label=r'Test data') 

    # # ax3.loglog(freqs1, psd1, 'r', alpha=0.5)
    # # ax4.loglog(freqs0, psd0, 'b', alpha=0.5)

    # ax3.loglog(freqs1, psd1, 'rosybrown', freqs0, psd0, 'slategray', alpha=0.7)
               
    # #ax2.plot(CD2, linewidth=2, color='rosybrown')
    
    cd1 =np.reshape(TestTargetVec, (115, 115))
    cd0 =np.reshape(TestClutterVec, (115, 115))
    
    
    
    cd1Avg=moving_average(cd1)
    cd0Avg =moving_average(cd0)
    
    
    
    # fig = plt.figure('test')
    # plt.suptitle('REF / TEST / Change')
    
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(ItestTarget, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(IrefTarget, cmap = plt.cm.gray)
    # plt.axis("off") #[180:245,170:235]
    
    # plt.show() 
    
    # print(g)
   


    # plt.show()      
    
    # fig, axhist = plt.subplots(1, 1, figsize=(8, 4))
    
 

    # # Red
    # show_hist(cd0, bins=200, histtype='step',
    #         lw=1, edgecolor= low, alpha=0.8, facecolor='r', ax=axhist)

    # # Green
    # show_hist(cd1, bins=200, histtype='step',
    #         lw=1, edgecolor= medium, alpha=0.8, facecolor='r', ax=axhist)

    # # Blue

    
    # #create legend

    # handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    # labels= ["clutter-clutter","target-clutter"]
    # plt.legend(handles, labels)
    

    
    # plt.show()




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
            
            ax1.plot(cd1Avg.ravel(), color='rosybrown', linewidth=2, label=r'target-clutter (Avg)') 
            ax1.plot(cd0Avg.ravel(), color='slategray', linewidth=2,  label=r'clutter-clutter (Avg)') 
            ax1.plot(CD1, color='red', linewidth=2, alpha=0.15, label=r'target-clutter')
            ax1.plot(CD0, color='gray', linewidth=2,alpha=0.15, label=r'clutter-clutter')
    
   
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
            ax2.set_prop_cycle(color=['slategray', 'rosybrown']) #linewidth=2
            
            #ax2.loglog(freqs1, psd1, 'rosybrown', freqs0, psd0, 'slategray', alpha=0.7)
            #ax2.loglog(freqs1, psd1, 'rosybrown', freqs0, psd0, 'slategray', alpha=0.7)
            # sns.kdeplot(np.array(CD0), color='slategray',linestyle='-.', ax=ax2) 
            # sns.kdeplot(np.array(CD1),color='rosybrown', linestyle='-.', ax=ax2)
            
            sns.kdeplot(np.array(CD0), color='gray', alpha=0.25,ax=ax2) 
            sns.kdeplot(np.array(CD1),color='red', alpha=0.25,ax=ax2) 
            
            sns.kdeplot(np.array(cd0Avg.ravel()), color='slategray', ax=ax2) 
            sns.kdeplot(np.array(cd1Avg.ravel()),color='rosybrown', ax=ax2) 
            
            
           # ax2.set(**pparam2)
            ax2.legend(prop={'size': 5})
            ax2.set_ylabel(r'Density', fontsize=10)  
            ax2.yaxis.set_label_position("right")
            ax2.set_xlabel(r'Magnitude', fontsize=10)  
            ax2.xaxis.set_label_coords(.5, -.2)
            ax2.yaxis.set_label_coords(-0.14, 0.5)
            #ax2.set_xlim([0, 1])
            ax2.yaxis.set_tick_params(labelleft=False, labelright=True)


            ax1.legend(loc='lower center', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=4)
            


            # ax1.text(0.5,-0.09, "(a)", size=12, ha="center", 
            #         transform=ax1.transAxes)
            # ax2.text(0.5,-0.09, "(b)", size=12, ha="center", 
            #         transform=ax2.transAxes)
    path='/home/marcello-costa/workspace/'
 
    fig.savefig(path + 'psdn2.png', dpi=DPI)
    fig.tight_layout()
    fig.tight_layout()
            
          

    plt.show()      

            
        
    
 












