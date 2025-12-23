# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:27:45 2024

@author: Dell
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio.functional as FF
from datetime import datetime
import random
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from scipy.stats import pearsonr, linregress
import time
import psutil


np.random.seed(1120)
torch.manual_seed(1120)
random.seed(1120)

def batch_pearsonr(tensor1, tensor2):
    batch_size = tensor1.shape[0]
    corr_coeffs = []
    for i in range(batch_size):
        # Extract the sequence of the current sample
        seq1 = tensor1[i].numpy()
        seq2 = tensor2[i].numpy()
        # Calculate the Pearson correlation coefficient of the current sample
        corr, _ = pearsonr(seq1, seq2)
        corr_coeffs.append(corr)
    # Calculate the average of the Pearson correlation coefficients of all samples
    average_corr = sum(corr_coeffs) / batch_size
    return average_corr

# Import utility functions
from ECG_tool import Compose_filter, sliding_window2, max_normalize_rows, max_min_normalize,z_normalize_rows

# Calculate RMSE
def rmse(predictions, labels):
    mse_loss = nn.MSELoss()
    mse = mse_loss(predictions, labels)
    return torch.sqrt(mse).item()

# Data preprocessing function
def data_pre(ABD5_path, ABD12_path, MASK_path, out_size, test_sample, bs):
    np.random.seed(1120)
    try:
        # Get the file list
        file_names = os.listdir(ABD5_path)
        file_names1 = os.listdir(ABD12_path)
        file_names2 = os.listdir(MASK_path)

        # Read ABD5 data
        wav_raw = []
        for file_name in file_names:
            name = file_name.split('.')[0]
            print(name)
            wav_path = os.path.join(ABD5_path, file_name)
            fh = pd.read_csv(wav_path).to_numpy()
            if np.isnan(fh).any():
                print(f"在文件 {file_name} 中发现了NaN值")
            wav_raw.append(fh)

        # Read ABD12 data
        wav_raw1 = []
        wav_raw2 = []
        for file_name1 in file_names1:
            name1 = file_name1.split('.')[0]
            print(name1)
            if len(name1) > 3:
                wav_path1 = os.path.join(ABD12_path, name1 + '.csv')
                fh1 = pd.read_csv(wav_path1).to_numpy()
                wav_raw1.append(fh1)
                wav_path2 = os.path.join(ABD12_path, name1[:-1] + '.csv')
                fh2 = pd.read_csv(wav_path2).to_numpy()
                fh3 = torch.tensor(fh2, dtype=torch.float).T
                fh2r = FF.resample(fh3, 500, 1000)
                wav_r = np.concatenate((fh1, fh2r.numpy().T), axis=1)
                wav_raw2.append(wav_r)

        # Select the sample dataset data
        sampleN = 5
        if sampleN == 17:
            ecg_raw = np.concatenate((wav_raw, wav_raw2))
        elif sampleN == 5:
            ecg_raw = np.array(wav_raw)
        elif sampleN == 12:
            ecg_raw = np.array(wav_raw2)

        print('sample number---', len(ecg_raw))
        print('now time is~', datetime.now().strftime("%m%d_%H%M%S"))

        # Eliminate baseline noise and power frequency interference
        fs = 1000
        ecg_1 = Compose_filter(ecg_raw, fs=fs, lowf=100, highf=1, notchf=50)

        # Divide the training, validation and test sets
        K = test_sample
        print('test sample', K, '/', sampleN)

        # Randomly select one from the remaining samples as the validation set
        remaining_indices = [i for i in range(len(ecg_1)) if i != K]
        val_sample = random.choice(remaining_indices)  # 随机选择验证集样本
        print('val sample', val_sample, '/', sampleN)

        # test
        ecg_1_test = ecg_1[K, :, :].reshape(1, -1, 5)

        # val
        ecg_1_val = ecg_1[val_sample, :, :].reshape(1, -1, 5)

        # train
        train_indices = [i for i in remaining_indices if i != val_sample]
        ecg_1_train = ecg_1[train_indices]

        ecg_1_train = np.transpose(ecg_1_train, axes=(0, 2, 1))
        ecg_1_val = np.transpose(ecg_1_val, axes=(0, 2, 1))
        ecg_1_test = np.transpose(ecg_1_test, axes=(0, 2, 1))

        # segment
        N_s = out_size
        N_w = N_s
        ecg_2_train = sliding_window2(ecg_1_train, window_size=N_s, step=N_w)
        ecg_2_val = sliding_window2(ecg_1_val, window_size=N_s, step=N_w)
        ecg_2_test = sliding_window2(ecg_1_test, window_size=N_s, step=N_w)

        # Process train
        ecg_3_train = np.zeros([1, 4, N_s])
        ecg_3_label = np.zeros([1, N_s])
        for s in range(len(ecg_2_train)):
            ecg_3 = np.array(ecg_2_train[s])
            ecg_3_train_slice = np.transpose(ecg_3, axes=(1, 0, 2))[:, 1:, :]
            ecg_3_train = np.vstack((ecg_3_train, ecg_3_train_slice))
            ecg_3_label_slice = np.transpose(ecg_3, axes=(1, 0, 2))[:, 0, :]
            ecg_3_label = np.vstack((ecg_3_label, ecg_3_label_slice))
        ecg_3_train = ecg_3_train[1:]
        ecg_3_label = ecg_3_label[1:]

        # Process val
        ecg_3_2 = np.array(ecg_2_val)
        ecg_3_val = np.transpose(ecg_3_2, axes=(0, 2, 1, 3))[:, :, 1:, :].reshape(-1, ecg_3_2.shape[-3] - 1,
                                                                                  ecg_3_2.shape[-1])
        ecg_3_label_val = np.transpose(ecg_3_2, axes=(0, 2, 1, 3))[:, :, 0, :].reshape(-1, ecg_3_2.shape[-1])

        # Process test
        ecg_3_1 = np.array(ecg_2_test)
        ecg_3_test = np.transpose(ecg_3_1, axes=(0, 2, 1, 3))[:, :, 1:, :].reshape(-1, ecg_3_1.shape[-3] - 1,
                                                                                   ecg_3_1.shape[-1])
        ecg_3_label2 = np.transpose(ecg_3_1, axes=(0, 2, 1, 3))[:, :, 0, :].reshape(-1, ecg_3_1.shape[-1])

        # normalization
        ecg_3_train1 = max_normalize_rows(ecg_3_train)
        ecg_3_val1 = max_normalize_rows(ecg_3_val)
        ecg_3_test1 = max_normalize_rows(ecg_3_test)
        ecg_3_label_tr = max_normalize_rows(ecg_3_label)
        ecg_3_label_val = max_normalize_rows(ecg_3_label_val)
        ecg_3_label_te = max_normalize_rows(ecg_3_label2)

        # Load the dataset
        data_set = TensorDataset(torch.tensor(ecg_3_train1, dtype=torch.float),
                                 torch.tensor(ecg_3_label_tr, dtype=torch.float))
        val_set = TensorDataset(torch.tensor(ecg_3_val1, dtype=torch.float),
                                torch.tensor(ecg_3_label_val, dtype=torch.float))
        test_set = TensorDataset(torch.tensor(ecg_3_test1, dtype=torch.float),
                                 torch.tensor(ecg_3_label_te, dtype=torch.float))

        train_loader = DataLoader(data_set, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

        return train_loader, val_loader, test_loader, test_sample
    except Exception as e:
        print(f"An error occurred during the data preprocessing process: {e}")
        return None, None, None, None

from thop import profile  # FLOPs
# ----------------------------------------------------
def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_tensor, device, repetitions=100):

    model.eval()
    input_tensor = input_tensor.to(device)


    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    cpu_mem_usage = []
    gpu_mem_usage = []
    total_time = 0.0

    # timing and memory measurement
    if device.type == 'cuda':
        # GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Reset the GPU memory statistics
        torch.cuda.reset_peak_memory_stats(device)
        start_event.record()

        with torch.no_grad():
            for _ in range(repetitions):
                model(input_tensor)
                # GPU memory usage（MB）
                current_gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                gpu_mem_usage.append(current_gpu_mem)
                # CPU memory usage（MB）
                current_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
                cpu_mem_usage.append(current_cpu_mem)

        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)  # total time（ms）

    else:
        # CPU time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(repetitions):
                model(input_tensor)
                # CPU memory usage（MB）
                current_cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
                cpu_mem_usage.append(current_cpu_mem)
        total_time = (time.time() - start_time) * 1000  #ms
        gpu_mem_usage = [0.0] * repetitions  # The GPU memory is 0 in CPU mode

    # calculate the AVG
    avg_inference_time = total_time / repetitions
    avg_cpu_mem = sum(cpu_mem_usage) / len(cpu_mem_usage) if cpu_mem_usage else 0.0
    avg_gpu_mem = sum(gpu_mem_usage) / len(gpu_mem_usage) if gpu_mem_usage else 0.0


    return {
        "inference_time_ms": avg_inference_time,
        "avg_cpu_mem_mb": avg_cpu_mem,
        "avg_gpu_mem_mb": avg_gpu_mem
    }


# train
def train(model_name, train_data, val_data, test_data, epochs, out_size, test_sample, save_path, Ch_defect_num1, Ch_defect_num2, Ch_defect1=False, Ch_defect2=False):

    K = test_sample
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    if model_name == 'Unet':
        from Unet_2 import Unet
        model = Unet(out_size)
        lr = 0.0001
    elif model_name == 'BiLSTM':
        from BiLSTM import BiLSTM
        model = BiLSTM(4, 128, 2, out_size=out_size)
        lr = 0.0001
    elif model_name == 'BiLSTM_S':
        from BiLSTM_s import CombinedLSTM
        model = CombinedLSTM(4, 128, 2)
        lr = 0.0001
    else:
        print("Unknown model name: ", model_name)
        return 0

    #Loss function, optimizer
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Model performance statistics

    input_shape = (1, 4, out_size)  # （batch=1）
    dummy_input = torch.randn(input_shape).to(device)  # FLOPs

    param_count = count_parameters(model)
    param_size_mb = param_count * 4 / (1024 ** 2)  # MB

    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    # （GFLOPs：10^9次）
    flops_gflops = flops / (10 ** 9)


    # Initialize the loss list
    train_loss = []
    Val_Loss = []
    BEST_LOSS = 100
    patience = 30  #
    patience_counter = 0  # Count the rounds that have not improved continuously
    early_stop = False  # Early stop sign

    for epoch in range(epochs):
        MODEL_SAVE_FLAG = 0
        run_loss = 0.0

        model.train()
        for batch_idx, (data, label) in enumerate(train_data):
            chs = []
            nums = list(range(4))
            random.shuffle(nums)
            if Ch_defect1:
                chs = nums[:Ch_defect_num1]
                #print('chs:',chs)
                for ch in chs:
                    data[:, ch, :] = 0
            #print(chs)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            if model_name == 'BiLSTM_S':
                output, alpha, l2_loss, channel_preds = model(data)

                loss1 = criterion(channel_preds[0], label)
                loss2 = criterion(channel_preds[1], label)
                loss3 = criterion(channel_preds[2], label)
                loss4 = criterion(channel_preds[3], label)
                loss0 = criterion(output, label)
                #print('loss', loss0, loss1, loss2, loss3, loss4)
                loss = loss0 +loss1 +loss2 +loss3 +loss4

            else:
                output = model(data)
                loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            #print('updated, batch_idx:{}'.format(batch_idx))
            run_loss += loss.item()

        # training loss
        epoch_loss = run_loss / len(train_data)
        train_loss.append(epoch_loss)
        print('====Epoch: {}, ===Train_Loss: {:.6f},+chs:{}'.format(epoch, epoch_loss,chs))

        # val
        model.eval()
        with torch.no_grad():
            run2_loss = 0.0
            for batch_idx1, (data1, label1) in enumerate(val_data):
                chs1 = []
                nums1 = list(range(4))
                random.shuffle(nums1)
                #print(nums1)
                if Ch_defect2:
                    chs1 = nums1[:Ch_defect_num2]
                    for ch1 in chs1:
                        data1[:, ch1, :] = 0

                data1 = data1.to(device)
                label1 = label1.to(device)
                if model_name == 'BiLSTM_S':
                    output1, alpha1, l2_loss1, _ = model(data1)
                else:
                    output1 = model(data1)

                loss1 = criterion(output1, label1)
                run2_loss += loss1.item()
            epoch_loss1 = run2_loss / len(val_data)
            Val_Loss.append(epoch_loss1)
            print('====Epoch: {}, ===Val_Loss: {:.6f},+chs1:{}'.format(epoch, epoch_loss1, chs1))

        # save model
        if epoch_loss1 < BEST_LOSS:
            BEST_LOSS = epoch_loss1
            patience_counter = 0  # reset
            # save best
            folder_path = save_path
            timestamp = datetime.now().strftime("%m%d_%H%M")
            model_state = str(Ch_defect1) + str(Ch_defect_num1) + str(Ch_defect2) + str(Ch_defect_num2)
            model_save_name = model_name + '_TS_' + str(K) + '_' + model_state + '.pth'
            model_save_path = os.path.join(folder_path, model_save_name)
            torch.save(model.state_dict(), model_save_path)
            print('[Model saved!]Best Loss: {:.6f}, ~RMSE: {:.6f}'.format(BEST_LOSS, BEST_LOSS ** 0.5))
        else:
            patience_counter += 1  # 验证损失未改善，计数器+1
            print(f'Verify that the loss has not improved and stop the counter early: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'continue{patience}has not improved，stop！')
                early_stop = True  # flag

    #test
    # load best
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    #test
    model.eval()
    metric_test = []
    test_losses = []

    #save waveform
    all_predictions = []
    all_labels = []


    for iter_idx in range(10):
        mae_list = []
        rmse_list = []
        pcc_list = []

        predictions = []
        labels = []

        with torch.no_grad():
            run2_loss = 0.0
            MAE_loss = 0.0
            RMSE_loss = 0.0
            PCC_loss = 0.0

            for batch_idx2, (data2, label2) in enumerate(test_data):
                chs2 = []
                nums2 = list(range(4))
                random.shuffle(nums2)
                if Ch_defect2:
                    chs2 = nums2[:Ch_defect_num2]

                    for ch2 in chs2:
                        data2[:, ch2, :] = 0

                data2 = data2.to(device)
                label2 = label2.to(device)
                if model_name == 'BiLSTM_S':
                    output2, alpha2, _, _ = model(data2)

                else:
                    output2 = model(data2)

                predictions.extend(output2.cpu().numpy())
                labels.extend(label2.cpu().numpy())

                loss2 = criterion(output2, label2)
                run2_loss += loss2.item()
                # MAE\RMSE\PCC
                loss2_1 = nn.L1Loss()(output2, label2)
                MAE_loss += loss2_1.item()

                rmse_value = rmse(output2, label2)
                RMSE_loss += rmse_value

                correlation,_ = pearsonr(output2.cpu().squeeze(), label2.cpu().squeeze())
                PCC_loss += correlation

            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            iter_loss = run2_loss / len(test_data)
            test_losses.append(iter_loss)
            mae_list.append(MAE_loss / len(test_data))
            rmse_list.append(RMSE_loss / len(test_data))
            pcc_list.append(PCC_loss / len(test_data))


            print('****test samlple:{}, *Iter: {}, ***test_Loss: {:.6f}, +chs2:{}'.format(K, iter_idx, iter_loss,chs2))


        # save metric
        name_str = '_' + str(K) + '_' + model_state + '_'
        csv_path1 = os.path.join(folder_path, model_name + name_str + timestamp + '.csv')

        iter_metrics = np.stack([mae_list, rmse_list, pcc_list], axis=0)
        print('MAE,RMSE,PCC', iter_metrics.T)
        metric_test.append(iter_metrics.squeeze())
        df2 = pd.DataFrame(metric_test)
        df2.to_csv(csv_path1, index=True)

        # -------------------------- inference time measurement--------------------------
        #  inference time and memory usage
        # inference_stats = {}
        test_input = None
        for data2, _ in test_data:
            test_input = data2
            break

        if test_input is not None:
            if test_input.shape[0] > 1:
                test_input = test_input[0:1]

            inference_stats = measure_inference_time(model, test_input, device)
        else:

            inference_stats = {
                "inference_time_ms": 0.0,
                "avg_cpu_mem_mb": 0.0,
                "avg_gpu_mem_mb": 0.0
            }
        # ----------------------------------------------------------------------------------
        # -------------------------- save to txt --------------------------
        # path
        os.makedirs(save_path, exist_ok=True)
        stats_file_path = os.path.join(save_path, f"{model_name}_stats.txt")

        # save to txt
        with open(stats_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(f"Model performance statistics report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Test sample number: {test_sample}\n")
            f.write(f"Input sequence length: {out_size}\n")
            f.write(f"Epoch: {epochs}\n")
            f.write("-" * 50 + "\n")
            f.write(f"param_count: {param_count:,} 个\n")
            f.write(f"param size(float32): {param_size_mb:.4f} MB\n")
            f.write(f"FLOPs: {flops_gflops:.4f} GFLOPs\n")
            f.write(f"Single inference time (batch_size=1): {inference_stats['inference_time_ms']:.4f} ms\n")
            f.write(f"CPU memory usage: {inference_stats['avg_cpu_mem_mb']:.4f} MB\n")
            f.write(f"GPU memory usage: {inference_stats['avg_gpu_mem_mb']:.4f} MB\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best val loss: {BEST_LOSS:.6f}\n")
            f.write(f"Best val RMSE: {BEST_LOSS ** 0.5:.6f}\n")
            f.write("=" * 50 + "\n")
        # ----------------------------------------------------------------------------------

    return metric_test, df2


if __name__ == '__main__':

    """ Create a directory. """
    path = './ABD5_CSV'
    path1 = './ABD12_CSV'
    path3 ='./ab0_test/'
    model_name = 'BiLSTM_S'
    out_size = 1024

    metrics =[]
    
    for D_num1 in range(4):
        if D_num1 == 0:
            Defect1 = False
        else:
            Defect1 = True
        for D_num2 in range(4):
            if D_num2 == 0:
                Defect2 = False
            else:
                Defect2 = True
                #D_num1 = 0   #The number of all zero channels
                #D_num2 = 0
            print('start', D_num1, D_num2)
            df_list = []
            for k in range(5):
                test_sample = k     #Each sample is subjected to a test set once, K-fold
                epochs = 200

                train_loader,val_loader,test_loader,test_sample\
                    = data_pre(path,path1,out_size,test_sample,bs=32)
                print('train start ~',test_sample)
                epoch_metric, df2 = train(model_name,train_loader,val_loader,test_loader,\
                       epochs=epochs,out_size=out_size,test_sample=test_sample,save_path=path3, \
                       Ch_defect_num1=D_num1, Ch_defect_num2=D_num2, \
                       Ch_defect1=Defect1, Ch_defect2=Defect2)

                df_list.append(df2)

            df3 = pd.concat(df_list, axis=1)
            timestamp1 = datetime.now().strftime("%m%d_%H")
            df3.to_csv(path3 + '_' + model_name + timestamp1 + '#' + str(D_num1) + '#' + str(D_num2) + '__all.csv')

            print('save done')



