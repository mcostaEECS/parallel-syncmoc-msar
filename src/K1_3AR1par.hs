--module ChangeDetection where

import ForSyDe.Shallow (signal, fromSignal)
import ForSyDe.Shallow.Core.Vector
import ForSyDe.Shallow
import System.IO
import Data.Complex
import Data.Word
import Data.Ord (comparing)
--import Data.List (findIndices, sort)
import Data.List (maximumBy, minimumBy, findIndices, sort)
import Statistics.Quantile
import Control.Concurrent
import Control.Parallel.Strategies
import System.IO (readFile)
import Data.Function (on)
import Control.Monad
import Data.Maybe
import Data.Vector.Unboxed qualified as U
import Data.Massiv.Array
import GHC.Conc (labelThread)
import Data.Time


-- First-order Autoregressive Model [AR(1)]: (ELEMENTARY MODEL)
---------------------------------------------------------------------------------------------------------------

arSystem :: Int -> Int ->Signal (ForSyDe.Shallow.Matrix Double) -> Signal (ForSyDe.Shallow.Matrix Double) -> ForSyDe.Shallow.Matrix Double
arSystem dimx dimy y_initial x_n = fromSignal out !! 0
   where 
    out 
   
     | rho > 0.5 =  zipWithSY addMatrix x' y_delayed'
     | rho > -0.5 =  zipWithSY addMatrix x' y_delayed'
     | otherwise         = x_n
         
    nrho = zipWithSY pearson (signal [x_n]) (signal [y_initial])
    rho = fromSignal nrho !! 0
    x'         = mapSY (scale rho) (x_n)
    y_delayed' = mapSY (scale' rho) y_delayed
    r = fromSignal y_initial !! 0
    y_delayed  = delaySY r out
   
 
    -- p-MC: MARKOV-CHAIN transition probabilities Pkj
---------------------------------------------------------------------------------------------------------------

mcSystem :: Int -> Int ->Signal (ForSyDe.Shallow.Matrix Double) -> Signal (ForSyDe.Shallow.Matrix Double) -> Double
mcSystem dimx dimy filtered ref = y_cd
   where y_cd = rho
         nrho = zipWithSY pearson (signal [filtered]) (signal [ref])  -- normalized sample cross-correlation
         rho = fromSignal nrho !! 0

        
         
-- Auxiliar: AR recursion
---------------------------------------------------------------------------------------------------------------

scale :: Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double
scale rho matrix = mapV (mapV f) matrix
  where f x = rho * x

scale' :: Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double  
scale' rho matrix = mapV (mapV f) matrix
  where f x = sqrt(1 - rho**2) * x

addMatrix :: ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double  
addMatrix a b = zipWithV (zipWithV (\x y -> x+y)) a b

subMatrix :: ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double  
subMatrix a b = zipWithV (zipWithV (\x y -> x-y)) a b

avgMatrix :: ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double  
avgMatrix a b = zipWithV (zipWithV (\x y -> (x+y)/2)) a b


pearson :: Signal (ForSyDe.Shallow.Matrix Double) -> Signal (ForSyDe.Shallow.Matrix Double) -> Double
pearson xs ys = (n * sumXY - sumX * sumY) / 
                 sqrt ( (n * sumX2 - sumX*sumX) * 
                        (n * sumY2 - sumY*sumY) ) 
  where  
        t = fromSignal xs !! 0
        r = fromSignal ys !! 0
        xn = flaTTen t
        yn = flaTTen r
        n = fromIntegral (length xn)
        sumX = Prelude.sum xn
        sumY = Prelude.sum yn
        sumX2 = Prelude.sum $ Prelude.zipWith (*) xn xn
        sumY2 = Prelude.sum $ Prelude.zipWith (*) yn yn
        sumXY = Prelude.sum $ Prelude.zipWith (*) xn yn



-- Auxiliar: Anomaly Detection
----------------------------------------------------------------------------------------------------------------
anomaly ::  Int -> Int -> Signal (ForSyDe.Shallow.Matrix Double) -> ForSyDe.Shallow.Matrix Double
anomaly dimx dimy ns = out
    where
        out =matrix dimx dimy $ Prelude.tail $ scanl (\acc n -> if n > hUP && n < hDOWN then 0  else n) 1 $ flaTTen dat
        dat = fromSignal ns !! 0
        a = flaTTen dat

        hUP = midspread normalUnbiased 3 (U.fromList a) 
        hDOWN = midspread normalUnbiased 4 (U.fromList a)


dotMatrix :: ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Double  
dotMatrix ns ms = zipWithMat (*) ns ms 

flaTTen :: ForSyDe.Shallow.Matrix a -> [a]
flaTTen = concatMap fromVector . fromVector



-- Auxiliar: Data reading
----------------------------------------------------------------------------------------------------------------

readDat :: String            -- ^ file content
        -> (Int, Int, [Double]) -- ^ (X dimension, Y dimension, list of pixel values)
readDat str = (dimX, dimY, image)
  where
    image = Prelude.map Prelude.read $ words str :: [Double]
    dimX = 500
    dimY = 500

partition :: Int -> Int -> [a] -> [[a]]
partition x y list
  | y - length image > 1 = error "image dimention Y mismatch"
  | otherwise = image
  where
    image = groupEvery x list
    
groupEvery n [] = []
groupEvery n l | length l < n = error "input Data is ill-formed"
                | otherwise    = Prelude.take n l : groupEvery n (Prelude.drop n l)

asciiLevels :: [Char]
asciiLevels = ['0','1',':','-','=','+','/','t','z','U','w','*','0','#','%','@']

toAsciiArt :: ForSyDe.Shallow.Matrix Double -> ForSyDe.Shallow.Matrix Char
toAsciiArt = mapMat num2char
    where
    num2char n = asciiLevels !! level n
    level n = truncate $ nLevels * (n / 255)
    nLevels = fromIntegral $ length asciiLevels - 1


chunks :: Int -> Int -> Signal (ForSyDe.Shallow.Matrix Double) -> Signal (ForSyDe.Shallow.Matrix (ForSyDe.Shallow.Matrix Double))
chunks dimx dimy img = mapSY (groupMat dimx dimy) img


-- Auxiliar: Stencil with Massiv
----------------------------------------------------------------------------------------------------------------
spatialFilter :: Int -> Int -> ForSyDe.Shallow.Matrix Double  -> ForSyDe.Shallow.Matrix Double  
spatialFilter dimx dimy img = matrix dimx dimy $ toList $ dropWindow $ mapStencil Edge (avgStencil 9) barImg
    where

        --y_n' = fromSignal img  !! 0
        
        imG = fromMatrix img
        
        barImg = fromLists' Seq [imG] :: Array U Ix2 Double

average3x3Filter :: Fractional a => Stencil Ix2 a a
average3x3Filter = makeStencil (Sz (3 :. 3)) (1 :. 1) $ \ get ->
    (  get (-1 :. -1) + get (-1 :. 0) + get (-1 :. 1) +
        get ( 0 :. -1) + get ( 0 :. 0) + get ( 0 :. 1) +
        get ( 1 :. -1) + get ( 1 :. 0) + get ( 1 :. 1)   ) / 9
{-# INLINE average3x3Filter #-}


---Sorting Methods

bubbleSort :: Ord a => [a] -> [a]
bubbleSort xs = foldr bubble [] xs
                where
                bubble x []     = [x]
                bubble x (y:ys) | x < y     = x:y:ys
                                | otherwise = y:bubble x ys


qsort  [] = []
qsort l@(x:xs) = qsort small ++ mid ++ qsort large
    where
        small = [y | y<-xs, y<x]
        mid   = [y | y<-l, y==x]
        large = [y | y<-xs, y>x]


reverseOrder :: [a] -> [a]
reverseOrder [] = []
reverseOrder (x : xs) = reverseOrder xs ++ [x]



-- Call functions
----------------------------------------------------------------------------------------------------------------
procMatrix3 :: Int -> Int -> ForSyDe.Shallow.Vector (Signal(ForSyDe.Shallow.Matrix (ForSyDe.Shallow.Matrix Double)))-> ForSyDe.Shallow.Matrix (ForSyDe.Shallow.Matrix Double) 
procMatrix3 dimx dimy dat = res
            where
    
            t = dat `atV` 0; st = fromSignal t !! 0
            
            ref1 = dat `atV` 1; sr1 = fromSignal ref1 !! 0; ssr1 = vector [sr1]
            ref2 = dat `atV` 3; sr2 = fromSignal ref2 !! 0; ssr2 = vector [sr2]
            ref3 = dat `atV` 5; sr3 = fromSignal ref3 !! 0; ssr3 = vector [sr3]
    
            sv = signal [ssr1, ssr2, ssr3]  
    
    
            m = (zipxSY . mapV (mapSY (zipWithMat(\ x y -> arSystem dimx dimy (signal [y]) (signal [x]) ) st )) . unzipxSY) sv
            c = (zipxSY . mapV (mapSY (zipWithMat(\ x y -> mcSystem dimx dimy (signal [y]) (signal [x]) ) st )) . unzipxSY) m
    
            
            p1 = fromSignal c !! 0; p1mat = fromMatrix p1 !! 0; p1List = p1mat `atV` 0
            p2 = fromSignal c !! 1; p2mat = fromMatrix p2 !! 0; p2List = p2mat `atV` 0
            p3 = fromSignal c !! 2; p3mat = fromMatrix p3 !! 0; p3List = p3mat `atV` 0
    
            pLists = [p1List] ++ [p2List] ++ [p3List]             
            
            m1 = fromSignal m !! 0; cm1 = m1 `atV` 0
            m2 = fromSignal m !! 1; cm2 = m2 `atV` 0
            m3 = fromSignal m !! 2; cm3 = m3 `atV` 0
    
            cms = [cm1, cm2, cm3]
            
            ---- Sorting Lists ---
            revpLists = reverseOrder pLists
            sorList = findIndices (\l -> l <= pLists !! 0) revpLists
    
            arCS = cms !! head sorList
            res = zipWithMat(\ x y -> subMatrix x y) st arCS

        
-- MAIN
----------------------------------------------------------------------------------------------------------------
main :: IO ()
main = do

    m1 <- newEmptyMVar
    m2 <- newEmptyMVar
    m3 <- newEmptyMVar
    m4 <- newEmptyMVar


    -- First HEC 
    test <- openFile "/home/marcello-costa/workspace/SPL/Data/test6/Itest6.dat" ReadMode; contentsTest <- hGetContents test

    ref1 <- openFile "/home/marcello-costa/workspace/SPL/Data/test6/Iref6B.dat" ReadMode; contentsRef1 <- hGetContents ref1
    ref3 <- openFile "/home/marcello-costa/workspace/SPL/Data/test6/Iref6I.dat" ReadMode; contentsRef3 <- hGetContents ref3
    ref5 <- openFile "/home/marcello-costa/workspace/SPL/Data/test6/Iref6M.dat" ReadMode; contentsRef5 <- hGetContents ref5
    
     
     ----- Dataset Arrangement---------------------------------------------------------------------------------------------

    let dimx =250; dimy = 250
        (dimX, dimY, imageStreamTest) = readDat contentsTest; intestMat = matrix dimX dimY imageStreamTest 
        (dimX1, dimY1, imageStreamRef1) = readDat contentsRef1; inrefMat1 = matrix dimX1 dimY1 imageStreamRef1
        (dimX3, dimY3, imageStreamRef3) = readDat contentsRef3; inrefMat3 = matrix dimX1 dimY1 imageStreamRef3
        (dimX5, dimY5, imageStreamRef5) = readDat contentsRef5; inrefMat5 = matrix dimX1 dimY1 imageStreamRef5
        
        
        st = signal [intestMat];  intest = mapSY (chunks dimx dimy)  (signal [st]) 
        sr1 = signal [inrefMat1]; inref1 = mapSY (chunks dimx dimy)  (signal [sr1])
        sr3 = signal [inrefMat3]; inref3 = mapSY (chunks dimx dimy)  (signal [sr3]) 
        sr5 = signal [inrefMat5]; inref5 = mapSY (chunks dimx dimy)  (signal [sr5])


    -- Test Sub-images
    let n = fromSignal m !! 0; m = fromSignal intest !! 0
        subImage1 = atMat 0 0 n; subImage2 = atMat 0 1 n; subImage3 = atMat 1 0 n; subImage4 = atMat 1 1 n
        st1 = signal [subImage1]; st2 = signal [subImage2]; st3 = signal [subImage3]; st4 = signal [subImage4]

    -- Ref Sub-Images
    let n = fromSignal m !! 0; m = fromSignal inref1 !! 0
        subRef1_1 = atMat 0 0 n; subRef1_2 = atMat 0 1 n; subRef1_3 = atMat 1 0 n; subRef1_4 = atMat 1 1 n
        sr1_1 = signal [subRef1_1]; sr1_2 = signal [subRef1_2]; sr1_3 = signal [subRef1_3]; sr1_4 = signal [subRef1_4]

    let n = fromSignal m !! 0; m = fromSignal inref3 !! 0
        subRef2_1 = atMat 0 0 n; subRef2_2 = atMat 0 1 n; subRef2_3 = atMat 1 0 n; subRef2_4 = atMat 1 1 n
        sr2_1 = signal [subRef2_1]; sr2_2 = signal [subRef2_2]; sr2_3 = signal [subRef2_3]; sr2_4 = signal [subRef2_4]

    let n = fromSignal m !! 0; m = fromSignal inref5 !! 0
        subRef3_1 = atMat 0 0 n; subRef3_2 = atMat 0 1 n; subRef3_3 = atMat 1 0 n; subRef3_4 = atMat 1 1 n
        sr3_1 = signal [subRef3_1]; sr3_2 = signal [subRef3_2]; sr3_3 = signal [subRef3_3]; sr3_4 = signal [subRef3_4]
    
  
   
     
     ----- Dataset Arrangement---------------------------------------------------------------------------------------------

    

    timeParallelStart <- getCurrentTime


    tid1 <- forkIO $ do

        myTid <- myThreadId
        labelThread myTid "parallelism 1"
        let dimx =250; dimy = 250

        let test1 = mapSY (chunks dimx dimy)  (signal [st1]) 
            ref1_1 = mapSY (chunks dimx dimy)  (signal [sr1_1])
            ref2_1 = mapSY (chunks dimx dimy)  (signal [sr2_1])
            ref3_1 = mapSY (chunks dimx dimy)  (signal [sr3_1])
            
    
        let u4_s1 = vector [test1,ref1_1,test1,ref2_1, test1,ref3_1]

        let m = zipWithxSY (procMatrix3 dimx dimy) u4_s1
            out = fromSignal m !! 0; mout = fromMatrix out !! 0
            sf = mapSY (spatialFilter dimx dimy) (signal [mout]) -- Spatial Filtering
            output = mapSY (anomaly dimx dimy) (signal [sf])      -- anomaly detection

        writeFile "/home/marcello-costa/workspace/SPL/Out/K1/Lag3/Par/proc1.txt" (show output)
        putMVar m1 output
        
        timeParallelEnd <- getCurrentTime
        putStrLn $ "parallelism 1 done, execution time: " ++ show(diffUTCTime timeParallelEnd timeParallelStart)


    tid2 <- forkIO $ do
        myTid <- myThreadId
        labelThread myTid "parallelism 2"
        let dimx =250; dimy = 250


        let test2 = mapSY (chunks dimx dimy)  (signal [st2]) 
            ref1_2 = mapSY (chunks dimx dimy)  (signal [sr1_2])
            ref2_2 = mapSY (chunks dimx dimy)  (signal [sr2_2])
            ref3_2 = mapSY (chunks dimx dimy)  (signal [sr3_2])
            
            


        let u4_s2 = vector [test2,ref1_2,test2,ref2_2, test2,ref3_2]


        let m = zipWithxSY (procMatrix3 dimx dimy) u4_s2
            out = fromSignal m !! 0; mout = fromMatrix out !! 0
            sf = mapSY (spatialFilter dimx dimy) (signal [mout]) -- Spatial Filtering
            output = mapSY (anomaly dimx dimy) (signal [sf])      -- anomaly detection
        writeFile "/home/marcello-costa/workspace/SPL/Out/K1/Lag3/Par/proc2.txt" (show output)
        putMVar m2 output
        
        timeParallelEnd <- getCurrentTime
        putStrLn $ "parallelism 2 done, execution time: " ++ show(diffUTCTime timeParallelEnd timeParallelStart)



    
    tid3 <- forkIO $ do
        myTid <- myThreadId
        labelThread myTid "parallelism 3"
        let dimx =250; dimy = 250
        let test3 = mapSY (chunks dimx dimy)  (signal [st3]) 
            ref1_3 = mapSY (chunks dimx dimy)  (signal [sr1_3])
            ref2_3 = mapSY (chunks dimx dimy)  (signal [sr2_3])
            ref3_3 = mapSY (chunks dimx dimy)  (signal [sr3_3])
      
        

        let u4_s3 = vector [test3,ref1_3,test3,ref2_3, test3,ref3_3]
        




        let m = zipWithxSY (procMatrix3 dimx dimy) u4_s3
            out = fromSignal m !! 0; mout = fromMatrix out !! 0
            sf = mapSY (spatialFilter dimx dimy) (signal [mout]) -- Spatial Filtering
            output = mapSY (anomaly dimx dimy) (signal [sf])      -- anomaly detection
        writeFile "/home/marcello-costa/workspace/SPL/Out/K1/Lag3/Par/proc3.txt" (show output)
        putMVar m3 output
        
        timeParallelEnd <- getCurrentTime
        putStrLn $ "parallelism 3 done, execution time: " ++ show(diffUTCTime timeParallelEnd timeParallelStart)


    tid4 <- forkIO $ do
        myTid <- myThreadId
        labelThread myTid "parallelism 4"
        let dimx =250; dimy = 250
        let test4 = mapSY (chunks dimx dimy)  (signal [st4]) 
            ref1_4 = mapSY (chunks dimx dimy)  (signal [sr1_4])
            ref2_4 = mapSY (chunks dimx dimy)  (signal [sr2_4])
            ref3_4 = mapSY (chunks dimx dimy)  (signal [sr3_4])
          
    
        let u4_s4 = vector [test4,ref1_4,test4,ref2_4, test4,ref3_4]

        let m = zipWithxSY (procMatrix3 dimx dimy) u4_s4
            out = fromSignal m !! 0; mout = fromMatrix out !! 0
            sf = mapSY (spatialFilter dimx dimy) (signal [mout]) -- Spatial Filtering
            output = mapSY (anomaly dimx dimy) (signal [sf])      -- anomaly detection
        writeFile "/home/marcello-costa/workspace/SPL/Out/K1/Lag3/Par/proc4.txt" (show output)
        putMVar m4 output
        
        timeParallelEnd <- getCurrentTime
        putStrLn $ "parallelism 4 done, execution time: " ++ show(diffUTCTime timeParallelEnd timeParallelStart)

    m1 <- takeMVar m1
    m2 <- takeMVar m2
    m3 <- takeMVar m3
    m4 <- takeMVar m4



    threadDelay 100

   
    -- -- let m = lengthS res
    -- -- print m



    -- -- ----- Output File ------------------------------------------------------------------------------------------------
    -- writeFile "/home/marcello-costa/workspace/Demos/OutP/K1/Lag3/CD0.txt" (show res)
    -- -- writeFile "/home/marcello-costa/workspace/Demos/Out/K1/Lag3/Test1.txt" (show intest)
    -- writeFile "/home/marcello-costa/workspace/Demos/Out/K1/Lag3/Iref2.txt" (show inref1) 
    
    -- RUN CODE USING THE TERMINAL :
    -- sudo apt install threadscope
    --  ghc -O2 Dai.hs -threaded -rtsopts -eventlog
    -- ./Dai Dai.1.txt +RTS -N2 -ls; threadscope Dai.eventlog


    










