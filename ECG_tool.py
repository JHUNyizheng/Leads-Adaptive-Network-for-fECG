# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:33:25 2024

@author: DELL
"""

import torch
from numpy.lib.stride_tricks import as_strided
 
def sliding_window(data,mask, window_size=1024, step=1024):
    # Create a sliding window view
    # Calculate the size of the matrix
    
    ecg_2_train = []
    #ecg_2_train_mask = []
    for i in range(data.shape[0]):
        ecg_ch =[]
        for j in range(data.shape[1]):
            #print('sample i',i,'ch j',j)
            rows = (data[i,j,:].size - window_size) // step + 1
            cols = window_size
            
            view = as_strided((data[i,j,:],), shape=(rows, cols),
                              strides=(step*data[i,j,:].itemsize, data[i,j,:].itemsize))
            
            view_mask = as_strided((mask[i,:],), shape=(rows, cols),
                              strides=(step*mask[i,:].itemsize, mask[i,:].itemsize))
            row_indices = np.where(view_mask.any(axis=1))[0]
            if row_indices.size >0:
                print('delete',len(row_indices),'-row',row_indices)
            view1 = np.delete(view, row_indices, axis=0)
            
            ecg_window = view1
            ecg_ch.append(ecg_window)
            
        ecg_2_train.append(ecg_ch)
    return ecg_2_train 

def sliding_window2(data, window_size=1024, step=1024):
    # Create a sliding window view
    # Calculate the size of the matrix
    
    ecg_2_train = []
    #ecg_2_train_mask = []
    for i in range(data.shape[0]):
        ecg_ch =[]
        for j in range(data.shape[1]):
            #print('sample i',i,'ch j',j)
            rows = (data[i,j,:].size - window_size) // step + 1
            cols = window_size
            
            view = as_strided((data[i,j,:],), shape=(rows, cols),
                              strides=(step*data[i,j,:].itemsize, data[i,j,:].itemsize))
            
           
            ecg_window = view
            ecg_ch.append(ecg_window)
            
        ecg_2_train.append(ecg_ch)
    return ecg_2_train 


def unroll_sliced_data(sliced_data, start_index, window_size, original_size):
    # Create an array full of zeros to store the restored data
    unrolled_data = [0] * original_size
    
    # Copy the sliced data to the correct location
    for i, value in enumerate(sliced_data):
        unrolled_data[i + start_index] = value
    
    return unrolled_data

def max_normalize_rows(matrix):
    if len(matrix.shape) == 3:
        normalized_matrix = np.empty_like(matrix)
        for s in range(matrix.shape[0]):
            for ch in range(matrix.shape[1]):
                maxvalue = np.max(np.abs(matrix[s,ch,:]))
                
                normalized_matrix[s,ch,:] = matrix[s,ch,:] / (maxvalue+1e-8)

                
        return normalized_matrix
    
    if len(matrix.shape) == 2:
        max_values = np.max(np.abs(matrix), axis=1, keepdims=True)
        normalized_matrix = matrix / max_values
        return normalized_matrix

def z_normalize_rows(matrix):
    if len(matrix.shape) == 3:
        normalized_matrix = np.empty_like(matrix)
        for s in range(matrix.shape[0]):
            for ch in range(matrix.shape[1]):
                mean_value = np.mean(matrix[s, ch, :])
                std_value = np.std(matrix[s, ch, :])
                normalized_matrix[s, ch, :] = (matrix[s, ch, :] - mean_value)/(std_value+1e-8)
        return normalized_matrix
    elif len(matrix.shape) == 2:
        mean_values = np.mean(matrix, axis = 1, keepdims = True)
        std_values = np.std(matrix, axis = 1, keepdims = True)
        normalized_matrix = (matrix - mean_values)/(std_values+1e-8)
        return normalized_matrix

def max_min_normalize(matrix):
    if len(matrix.shape) == 3:
        min_value = np.min(matrix)
        max_value = np.max(matrix)
        normalized_matrix = (matrix - min_value) / (max_value - min_value + 1e-8)
        return normalized_matrix
    elif len(matrix.shape) == 2:
        min_value = np.min(matrix)
        max_value = np.max(matrix)
        normalized_matrix = (matrix - min_value) / (max_value - min_value + 1e-8)
        return normalized_matrix

#The waveform is normalized by the maximum amplitude, the tensor object
def normalize_wave(waveform):
    
    max_amp = waveform.abs().max()
    normalized_wave = waveform / max_amp
    return normalized_wave

# EXAMPLES
#sliced_data = [1, 2, 3, 4, 5]  # Suppose this is the data from the sliding window slice
#start_index = 1  # Suppose the position where the sliding window starts
#window_size = 3  # Suppose the size of the sliding window
#original_size = 10  # Suppose the size of the original data
 
# Calling function
#unrolled_data = unroll_sliced_data(sliced_data, start_index, window_size, original_size)
 
#print(unrolled_data)


import numpy as np
from scipy import signal
#filter
def Compose_filter(ecg_raw, fs=1000, lowf=100,highf=1,notchf=50):
    Nyquist=0.5*fs
    low =lowf/Nyquist
    b1, a1 = signal.butter(4, low, 'lowpass')
    high =highf/Nyquist
    b2, a2 = signal.butter(4, high, 'highpass')
    w0 = notchf/(0.5*fs)
    b3,a3 = signal.iirnotch(w0, 50/0.1) #w0 standardized frequency (0.5 of fs is 1)
    ecg_1 = np.empty_like(ecg_raw)  # sample x signal x ch
    for i in range(len(ecg_raw)):
        for j in range(ecg_raw.shape[2]):
            ecg_filter1 = signal.filtfilt(b1, a1, ecg_raw[i,:,j]) 
            ecg_filter2 = signal.filtfilt(b3, a3, ecg_filter1)
            ecg_filter3 = signal.filtfilt(b2, a2, ecg_filter2)
            ecg_1[i,:,j] = ecg_filter3 * 1  # 10 uV 
    return ecg_1


import numpy as np
from scipy import signal


def mse_loss_torch_batch(A_batch, B_batch):
    mse_loss_batch = torch.mean((A_batch - B_batch) ** 2, dim=1)

    return mse_loss_batch


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True