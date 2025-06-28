import scipy.io as sio


def load_data(test_type):

    path = '/home/marcello-costa/workspace/SPL/Data/Dataset/'
    path2 = '/home/marcello-costa/workspace/SPL/Data/Dataset/targets/'
    
    # images of mission by pair
    ImageRead1=sio.loadmat(path+'ImageRead1.mat')
    Im1=ImageRead1['ImageRead1']
    ImageRead2=sio.loadmat(path+'ImageRead2.mat')
    Im2=ImageRead2['ImageRead2']
    ImageRead3=sio.loadmat(path+'ImageRead3.mat')
    Im3=ImageRead3['ImageRead3']
    ImageRead4=sio.loadmat(path+'ImageRead4.mat')
    Im4=ImageRead4['ImageRead4']
    ImageRead5=sio.loadmat(path+'ImageRead5.mat')
    Im5=ImageRead5['ImageRead5']
    ImageRead6=sio.loadmat(path+'ImageRead6.mat')
    Im6=ImageRead6['ImageRead6']
    ImageRead7=sio.loadmat(path+'ImageRead7.mat')
    Im7=ImageRead7['ImageRead7']
    ImageRead8=sio.loadmat(path+'ImageRead8.mat')
    Im8=ImageRead8['ImageRead8']
    ImageRead9=sio.loadmat(path+'ImageRead9.mat')
    Im9=ImageRead9['ImageRead9']
    ImageRead10=sio.loadmat(path+'ImageRead10.mat')
    Im10=ImageRead10['ImageRead10']
    ImageRead11=sio.loadmat(path+'ImageRead11.mat')
    Im11=ImageRead11['ImageRead11']
    ImageRead12=sio.loadmat(path+'ImageRead12.mat')
    Im12=ImageRead12['ImageRead12']
    ImageRead13=sio.loadmat(path+'ImageRead13.mat')
    Im13=ImageRead13['ImageRead13']
    ImageRead14=sio.loadmat(path+'ImageRead14.mat')
    Im14=ImageRead14['ImageRead14']
    ImageRead15=sio.loadmat(path+'ImageRead15.mat')
    Im15=ImageRead15['ImageRead15']
    ImageRead16=sio.loadmat(path+'ImageRead16.mat')
    Im16=ImageRead16['ImageRead16']
    ImageRead17=sio.loadmat(path+'ImageRead17.mat')
    Im17=ImageRead17['ImageRead17']
    ImageRead18=sio.loadmat(path+'ImageRead18.mat')
    Im18=ImageRead18['ImageRead18']
    ImageRead19=sio.loadmat(path+'ImageRead19.mat')
    Im19=ImageRead19['ImageRead19']
    ImageRead20=sio.loadmat(path+'ImageRead20.mat')
    Im20=ImageRead20['ImageRead20']
    ImageRead21=sio.loadmat(path+'ImageRead21.mat')
    Im21=ImageRead21['ImageRead21']
    ImageRead22=sio.loadmat(path+'ImageRead22.mat')
    Im22=ImageRead22['ImageRead22']
    ImageRead23=sio.loadmat(path+'ImageRead23.mat')
    Im23=ImageRead23['ImageRead23']
    ImageRead24=sio.loadmat(path+'ImageRead24.mat')
    Im24=ImageRead24['ImageRead24']
    
    
    # tragets position by mission
    tp1=sio.loadmat(path2+'S1.mat')
    tp1=tp1['S1']
    tp2=sio.loadmat(path2+'K1.mat')
    tp2=tp2['K1']
    tp3=sio.loadmat(path2+'F1.mat')
    tp3=tp3['F1']
    tp4=sio.loadmat(path2+'AF1.mat')
    tp4=tp4['AF1']
    
    if test_type == 'ARn1' or  test_type == 'ARn2' or  test_type == 'ARn3':
        
          par=[[Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, Im2,Im3,Im4, Im5, Im6, tp1,'S1', Im1, 1],
               [Im14, Im7,Im8,Im9,Im10, Im11, Im12, Im13,  Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, Im1,Im3,Im4, Im5, Im6,tp1,'S1', Im2, 2],
               [Im21, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20,  Im22, Im23, Im24,Im2,Im1,Im4, Im5, Im6, tp1,'S1', Im3, 3],
               [Im10, Im7,Im8,Im9, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, Im2,Im3,Im1, Im5, Im6,tp1,'S1', Im4, 4],
               [Im17, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16,  Im18, Im19, Im20, Im21, Im22, Im23, Im24,Im2,Im3,Im4, Im1, Im6, tp1,'S1', Im5, 5],
               [Im24, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23,Im2,Im3,Im4, Im5, Im1, tp1,'S1', Im6, 6],
               [Im13, Im1,Im2,Im3,Im4, Im5, Im6, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, Im8,Im9,Im10, Im11, Im12, tp2,'K1', Im7, 7],
               [Im20, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im21, Im22, Im23, Im24, Im7,Im9,Im10, Im11, Im12,tp2,'K1', Im8, 8],
               [Im3, Im1,Im2,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24,Im8,Im7,Im10, Im11, Im12, tp2,'K1', Im9, 9],
               [Im16, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15,  Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, Im8,Im9,Im7, Im11, Im12,tp2,'K1', Im10, 10],
               [Im23, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22,  Im24, Im8,Im9,Im10, Im7, Im12,tp2,'K1', Im11, 11],
               [Im6, Im1,Im2,Im3,Im4, Im5,  Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24,Im8,Im9,Im10, Im11, Im7, tp2,'K1', Im12, 12],
               [Im19, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im17, Im18, tp3,'F1', Im13, 13],
               [Im2,Im1, Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im13, Im15, Im16, Im17, Im18,tp3,'F1', Im14, 14],
               [Im9, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8,  Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im13, Im16, Im17, Im18,tp3,'F1', Im15,15],
               [Im22, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21,  Im23, Im24,Im14, Im15, Im13, Im17, Im18, tp3,'F1', Im16, 16],
               [Im5, Im1,Im2,Im3,Im4,  Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im13, Im18,tp3,'F1', Im17, 17],
               [Im12, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11,  Im19, Im20, Im21, Im22, Im23, Im24, Im14, Im15, Im16, Im17, Im13,tp3,'F1', Im18, 28],
               [Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im20, Im21, Im22, Im23, Im24,tp4,'AF1', Im19, 19],   
               [Im8, Im1,Im2,Im3,Im4, Im5, Im6, Im7,   Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im21, Im22, Im23, Im24, tp4,'AF1', Im20,20],
               [Im15, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14,  Im16, Im17, Im18, Im20, Im19, Im22, Im23, Im24,tp4,'AF1', Im21,21],
               [Im4, Im1,Im2,Im3, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im20, Im21, Im19, Im23, Im24, tp4,'AF1', Im22, 22],
               [Im11, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10,  Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im20, Im21, Im22, Im19, Im24,tp4,'AF1', Im23, 23],
               [Im18, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im20, Im21, Im22, Im23, Im19, tp4,'AF1', Im24, 24]]
    

    else:
        
        print('Invalid test')
    
        
        
            
    
        
    return par
    

    
    
