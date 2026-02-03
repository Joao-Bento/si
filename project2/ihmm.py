import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import medfilt
import os
import ivt

np.random.seed(42)

class eye_movement_detection:
    def __init__(self, df, frequency, fov_x, fov_y, v_threshold):
        self.df = df
        self.frequency = frequency
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.v_threshold = v_threshold
        
        emd_ivt = ivt.eye_movement_detection(df, frequency, fov_x, fov_y, v_threshold)
        observations = emd_ivt.observations
        observations = observations.reset_index(drop=True).astype(int)
        
        P, O, Miu = self.BaumWelch(observations, n_reestimations=4)
        estimations = self.Viterbi(observations, P, O, Miu)
        self.classifier(estimations)

    # Baum-Welch reestimation
    def BaumWelch(self, observations, n_reestimations=5):
        P = np.random.rand(2, 2)
        P = P / P.sum(axis=1, keepdims=True)
        O = np.random.rand(2, 2)
        O = O / O.sum(axis=1, keepdims=True)
        miu0 = np.random.rand(2)
        miu0 = miu0 / miu0.sum()
        #n_reestimations=1
        P_matrices = np.zeros((n_reestimations, len(miu0), len(miu0)))
        O_matrices = np.zeros((n_reestimations, len(miu0), len(miu0)))
        # miu0 = [.5, .5] #initial distribution
        # P = np.array([[.5, .5], [.5, .5]]) #transition probability matrix
        # O = np.array([[.9, .1], [.1, .9]]) #observation probability matrix
        for k in range(n_reestimations): #number of 
            Alpha = np.zeros((len(observations), len(miu0)))
            Beta = np.zeros((len(observations), len(miu0)))
            Miu = np.zeros((len(observations), len(miu0)))
            Miu[0] = miu0
            Ksi = np.zeros((len(observations), len(miu0), len(miu0)))
                
            #Forward-Backward algorithm
            alpha = np.diag(O[:,observations[0]]) @ np.array(miu0).T
            beta = np.ones(len(miu0))
            Alpha[0] = alpha 
            Beta[-1] = beta 
            for tau in range(1,len(observations)):
                zt = observations[tau]
                alpha = np.diag(O[:,zt]) @ (P.T @ alpha)
                alpha = alpha / alpha.sum()
                Alpha[tau] = alpha
            for tau in range(len(observations)-2, -1, -1):
                zt1 = observations[tau+1]
                beta = P @ np.diag(O[:,zt1]) @ beta
                beta = beta / beta.sum()
                Beta[tau] = beta
                
            for t in range(len(observations)):
                Miu[t] = (Alpha[t] * Beta[t]) / (Alpha[t].T @ Beta[t])
            for t in range(1,len(observations)):
                ksi = np.zeros((len(miu0), len(miu0)))
                for x1 in range(len(miu0)):
                    for x2 in range(len(miu0)):
                        ksi[x1, x2] = Alpha[t-1][x1] * Beta[t][x2] * P[x1, x2] * O[x2,observations[t]]
                ksi = ksi / ksi.sum()
                Ksi[t] = ksi

            #M-step
            for x1 in range(len(miu0)):
                for x2 in range(len(miu0)):
                    P[x1, x2] = Ksi[1:3, x1, x2].sum() / Miu[:2, x1].sum()
                for z in range(len(miu0)):
                    O[x1, z] = Miu[observations == z, x1].sum() / Miu[:, x1].sum()
                    
            #P = P_matrices[:k+1].mean(axis=0)
            #O = O_matrices[:k+1].mean(axis=0)
            # P = np.clip(P, 0.1, 0.9)
            # P = P / P.sum(axis=1, keepdims=True)
            # O = np.clip(O, 0.1, 0.9)
            # O = O / O.sum(axis=1, keepdims=True)
            P_matrices[k]=P
            O_matrices[k]=O
        return P, O, Miu

    

    

            
# print("Reestimated P:")
# print(P)
# print("Reestimated O:")
# print(O)  

# # Check if the sum of matrices P and O are 1
# print("Sum of rows in P:")
# print(P.sum(axis=1))
# print("Sum of rows in O:")
# print(O.sum(axis=1))

    # Viterbi sampler
    def Viterbi(self, observations, P, O, Miu):
        estimations = np.zeros(len(observations))
        # miu0 = [.5, .5] #initial distribution
        # P = np.array([[.5, .5], [.5, .5]]) #transition probability matrix
        # O = np.array([[1, 0], [0, 1]]) #observation probability matrix
        miu0 = Miu[0]
        mt = np.diag(O[:,observations[0]]) @ np.array(miu0).T
        mt = mt/mt.sum()
        I = np.zeros((len(observations), len(miu0)))
        for t in range(1, len(observations)):
            zt = observations[t]
            it = np.argmax((P.T @ np.diag(mt)), axis=1)
            mt = np.diag(O[:,zt]) @ np.max((P.T @ np.diag(mt)),axis=1)
            mt = mt/mt.sum()
            I[t] = it
        xt = np.argmax(mt)
        estimations[2] = xt
        for t in range(len(estimations)-2, -1, -1):
            estimations[t] = I[t+1, int(estimations[t+1])]
        return estimations

#df['estimations'] = estimations
    def classifier(self, estimations):
        self.df['fixations'] = estimations
        irwin = 0.15
        self.df['prediction'] = False
        fixations = []
        f=0
        while f < len(self.df):
            if self.df['fixations'].iloc[f]:
                fixation_start = f#self.df['ef_time'].iloc[f]
                while self.df['fixations'].iloc[f] and f < len(self.df)-1:
                    f += 1
                fixation_end = f-1#self.df['ef_time'].iloc[f]
                if self.df['time_seconds'].iloc[fixation_end] - self.df['time_seconds'].iloc[fixation_start] >= irwin: #Irwin 1992
                    fixations.append((fixation_start, fixation_end))
                    self.df.loc[fixation_start:fixation_end, 'prediction'] = True
                #print("Detected fixations", (fixation_start, fixation_end))
            f += 1
            
        self.FF = len(fixations)/(self.df['time_seconds'].iloc[-1]-self.df['time_seconds'].iloc[0])*60
        
        fixation_time = [(self.df['time_seconds'].iloc[start], self.df['time_seconds'].iloc[end]) for start, end in fixations]
        self.AFD = np.mean([end - start for start, end in fixation_time])


# print(f'Experimental number of fixations: {len(fixations)}')
# #print(f'Ground truth number of fixations: {len(df_gt)}')
# print(f'Experimental average fixation duration: {avg_fixation_duration:.3f} seconds')
# #print(f'Ground truth average fixation duration: {avg_fixation_duration_gt:.3f} seconds')
# print(f"Percentage of fixations: {(df['prediction'].sum()/len(df))*100:.2f}%")

