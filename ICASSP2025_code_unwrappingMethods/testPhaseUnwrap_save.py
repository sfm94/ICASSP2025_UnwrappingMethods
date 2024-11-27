import numpy as np
import math
from matplotlib import pyplot as plt
import scipy

from scipy.fftpack import fft, ifft
from scipy.fftpack import fftshift, ifftshift

from scipy.signal import find_peaks

from scipy.signal import butter, lfilter, freqz
from scipy .signal import firwin, remez,kaiser_atten, kaiser_beta
import scipy.signal as sig
import Functions_save as fun
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
StopTime = 20
bio_model = 'sin'#['sin','Albanese']''
Fc_br = 0.4
Fc_hr = 1.7
lambda_val = 3e8/60e9
Amp_br = 1 * lambda_val
Amp_hr = Amp_br/20
rbm_duration = int(StopTime/2)#15
F_rbm = 2.5
rbm_type = ['None','ramp','sin','general','comb_general']
is_rbm = 'Yes'
Max_N_K = 5
coeff = 0.6
low_breathing = 0.2
high_breathing = 0.7
low_heart = 0.8
high_heart = 4
distance_peaks_breathing = 0.05
distance_peaks_heart = 0.1
snr_dB = np.array([10,20,60])
SNR = 10**(snr_dB/10)
epsilon = 0.00000000001

MC_flag = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Monte Carlo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if MC_flag:
    length_Fs = 50
    length_rbm = 40
    Fs_all = np.linspace(10, 100.0, num=length_Fs)
    Amp_range = np.linspace(0, 100, num=length_rbm)*lambda_val
    shape = (len(Amp_range), len(Fs_all))

else:
    Fs_all = [100]
    Amp_range = [10*Amp_br]
    shape = (1,1)
    length_Fs = 1
    length_rbm = 1




error_peak_gt_MDACM_br = np.ones(shape)
error_peak_gt_unwrap1_br = np.ones(shape)
error_peak_gt_unwrap_python_br = np.ones(shape)
error_peak_gt_EATAN1_br = np.ones(shape)
error_peak_gt_EATAN2_br= np.ones(shape)
error_peak_gt_unlimited_br = np.ones(shape)

error_peak_gt_MDACM_hr = np.ones(shape)
error_peak_gt_unwrap1_hr = np.ones(shape)
error_peak_gt_unwrap_python_hr = np.ones(shape)
error_peak_gt_EATAN1_hr = np.ones(shape)
error_peak_gt_EATAN2_hr= np.ones(shape)
error_peak_gt_unlimited_hr = np.ones(shape)

error_gt_MDACM = np.ones(shape)
error_gt_unwrap1 = np.ones(shape)
error_gt_unwrap_python =np.ones(shape)
error_gt_EATAN1=np.ones(shape)
error_gt_EATAN2 =np.ones(shape)
error_gt_unlimited=np.ones(shape)

for k in range(len(rbm_type)):
    for snr_element in snr_dB:
        for row in range(length_rbm):
            rbm_amp = Amp_range[row]
            for col in range(length_Fs):
                Fs = Fs_all[col]
                nyquist_freq = Fs / 2
                filter_order = int(Fs) + 1

                dt = 1/Fs
                t = np.arange(np.round(StopTime / dt)) * dt
                x_rbm = fun.random_body_generation(t,dt,StopTime, rbm_duration,rbm_amp, rbm_type[k], F_rbm, phi=np.pi/3, BW=4 )
                x, x_without_rbm = fun.get_total_signal(x_rbm, t, Amp_br, Amp_hr, Fc_br, Fc_hr, Fs, StopTime, coeff, bio_model)
                ground_truth_signal = 4*np.pi*x/lambda_val
                expo_x_total = np.exp(1j * ground_truth_signal)
                SNR = 10 ** (snr_element / 10)
                expo_x_total+=(1/np.sqrt(SNR))*(np.random.random(size=expo_x_total.shape)+1j*np.random.random(size=expo_x_total.shape))
                N_fft = 4*len(expo_x_total)
                f_axis = np.linspace(0,Fs,N_fft)
                low_freq_breathing = low_breathing / nyquist_freq
                high_freq_breathing =high_breathing / nyquist_freq
                low_freq_heart= low_heart / nyquist_freq
                high_freq_heart = high_heart / nyquist_freq

                breathing_b = firwin(filter_order, [low_freq_breathing, high_freq_breathing],pass_zero=False)
                breathing_a = 1
                heart_b = firwin(filter_order, [low_freq_heart, high_freq_heart],pass_zero=False)
                heart_a = 1
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                filtered_breathing_GroundTruth = lfilter(breathing_b, breathing_a, ground_truth_signal)
                filtered_heart_GroundTruth = lfilter(heart_b, heart_a, ground_truth_signal)


                MDACM_signal = fun.MDACM(expo_x_total)
                filtered_breathing_MDACM = lfilter(breathing_b, breathing_a, MDACM_signal)
                filtered_heart_MDACM = lfilter(heart_b, heart_a, MDACM_signal)

                unwrap1_signal = fun.phase_unwrap1(expo_x_total)
                filtered_breathing_phase_unwrap1 = lfilter(breathing_b, breathing_a,unwrap1_signal )
                filtered_heart_phase_unwrap1 = lfilter(heart_b, heart_a, unwrap1_signal)

                phase_unwrap_python_signal = fun.phase_unwrap_python(expo_x_total)
                filtered_breathing_unwrap_python = lfilter(breathing_b, breathing_a, phase_unwrap_python_signal)
                filtered_heart_unwrap_python = lfilter(heart_b, heart_a, phase_unwrap_python_signal)

                EATAN1_signal = fun.EATAN1(expo_x_total)
                filtered_breathing_EATAN1 = lfilter(breathing_b, breathing_a,EATAN1_signal )
                filtered_heart_EATAN1 = lfilter(heart_b, heart_a, EATAN1_signal)


                EATAN2_signal = fun.EATAN2(expo_x_total,ground_truth_signal,t,lambda_val,Max_N_K)
                filtered_breathing_EATAN2 = lfilter(breathing_b, breathing_a, EATAN2_signal)
                filtered_heart_EATAN2 = lfilter(heart_b, heart_a, EATAN2_signal)

                phase_unlimited = fun.Unlimited_Sampling(expo_x_total,Fs, 1*np.ceil(np.max(abs(2*0.5/2/lambda_val*np.pi)) / (2*np.pi) )* 2 * np.pi, lambda_val,Max_N_K) # Br is a limitation of this method since it has a direct impact on N.
                filtered_breathing_unlimited = lfilter(breathing_b, breathing_a, phase_unlimited)
                filtered_heart_unlimited = lfilter(heart_b, heart_a, phase_unlimited)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                fft_windowed_signal_GroundTruth_breathing = np.abs(fft(np.multiply(filtered_breathing_GroundTruth,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_MDACM_breathing = np.abs(fft(np.multiply(filtered_breathing_MDACM,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_phase_unwrap1_breathing = np.abs(fft(np.multiply(filtered_breathing_phase_unwrap1,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_unwrap_python_breathing = np.abs(fft(np.multiply(filtered_breathing_unwrap_python,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_EATAN1_breathing = np.abs(fft(np.multiply(filtered_breathing_EATAN1,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_EATAN2_breathing = np.abs(fft(np.multiply(filtered_breathing_EATAN2,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_phase_unlimited_breathing = np.abs(fft(np.multiply(filtered_breathing_unlimited,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                fft_windowed_signal_GroundTruth_heart = np.abs(fft(np.multiply(filtered_heart_GroundTruth,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_MDACM_heart = np.abs(fft(np.multiply(filtered_heart_MDACM,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_phase_unwrap1_heart = np.abs(fft(np.multiply(filtered_heart_phase_unwrap1,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_unwrap_python_heart = np.abs(fft(np.multiply(filtered_heart_unwrap_python,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_EATAN1_heart = np.abs(fft(np.multiply(filtered_heart_EATAN1,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_EATAN2_heart = np.abs(fft(np.multiply(filtered_heart_EATAN2,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                fft_windowed_signal_phase_unlimited_heart = np.abs(fft(np.multiply(filtered_heart_unlimited,fun.signal.windows.blackmanharris(len(ground_truth_signal))),N_fft))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                interval_breathing = np.array(np.where((f_axis > low_breathing) & (f_axis < high_breathing)))[0]
                interval_heart = np.array(np.where((f_axis > low_heart) & (f_axis < high_heart)))[0]

                interval_breathing_gt = np.array(np.where((f_axis > Fc_br-0.1) & (f_axis < Fc_br+0.1)))[0]
                interval_heart_gt = np.array(np.where((f_axis > Fc_hr-0.1) & (f_axis < Fc_hr+0.1)))[0]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                peak_groundTruth_br, best_peak_index_groundTruth_br, filtered_peaks_groundTruth_br = fun.mypeaks(fft_windowed_signal_GroundTruth_breathing,interval_breathing_gt,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_MDACM_br, best_peak_index_MDACM_br, filtered_peaks_MDACM_br = fun.mypeaks(fft_windowed_signal_MDACM_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_unwrap1_br, best_peak_index_unwrap1_br, filtered_peaks_unwrap1_br = fun.mypeaks(fft_windowed_signal_phase_unwrap1_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_unwrap_python_br, best_peak_index_unwrap_python_br, filtered_peaks_unwrap_python_br = fun.mypeaks(fft_windowed_signal_unwrap_python_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_EATAN1_br, best_peak_index_EATAN1_br, filtered_peaks_EATAN1_br = fun.mypeaks(fft_windowed_signal_EATAN1_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_EATAN2_br, best_peak_index_EATAN2_br, filtered_peaks_EATAN2_br = fun.mypeaks(fft_windowed_signal_EATAN2_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                peak_unlimited_br, best_peak_index_unlimited_br, filtered_peaks_unlimited_br = fun.mypeaks(fft_windowed_signal_phase_unlimited_breathing,interval_breathing,distance_peaks_breathing,N_fft,Fs,f_axis)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                peak_groundTruth_hr, best_peak_index_groundTruth_hr, filtered_peaks_groundTruth_hr = fun.mypeaks(fft_windowed_signal_GroundTruth_heart,interval_heart_gt,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_MDACM_hr, best_peak_index_MDACM_hr, filtered_peaks_MDACM_hr = fun.mypeaks(fft_windowed_signal_MDACM_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_unwrap1_hr, best_peak_index_unwrap1_hr, filtered_peaks_unwrap1_hr = fun.mypeaks(fft_windowed_signal_phase_unwrap1_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_unwrap_python_hr, best_peak_index_unwrap_python_hr, filtered_peaks_unwrap_python_hr = fun.mypeaks(fft_windowed_signal_unwrap_python_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_EATAN1_hr, best_peak_index_EATAN1_hr, filtered_peaks_EATAN1_hr = fun.mypeaks(fft_windowed_signal_EATAN1_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_EATAN2_hr, best_peak_index_EATAN2_hr, filtered_peaks_EATAN2_hr = fun.mypeaks(fft_windowed_signal_EATAN2_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                peak_unlimited_hr, best_peak_index_unlimited_hr, filtered_peaks_unlimited_hr = fun.mypeaks(fft_windowed_signal_phase_unlimited_heart,interval_heart,distance_peaks_heart,N_fft,Fs,f_axis)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                error_gt_MDACM[row,col] = np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-MDACM_signal+np.mean(MDACM_signal))/len(ground_truth_signal))
                error_gt_unwrap1[row,col] = np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-unwrap1_signal+np.mean(unwrap1_signal))/len(ground_truth_signal))
                error_gt_unwrap_python[row,col] =np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-phase_unwrap_python_signal+np.mean(phase_unwrap_python_signal))/len(ground_truth_signal))
                error_gt_EATAN1[row,col] =np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-EATAN1_signal+np.mean(EATAN1_signal))/len(ground_truth_signal))
                error_gt_EATAN2[row,col] =np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-EATAN2_signal+np.mean(EATAN2_signal))/len(ground_truth_signal))
                error_gt_unlimited[row,col] =np.linalg.norm((ground_truth_signal-np.mean(ground_truth_signal)-phase_unlimited+np.mean(phase_unlimited))/len(ground_truth_signal))

                if  peak_MDACM_br  :
                    error_peak_gt_MDACM_br[row,col] = np.linalg.norm(peak_groundTruth_br-peak_MDACM_br)
                if peak_unwrap1_br  :
                    error_peak_gt_unwrap1_br[row,col] = np.linalg.norm(peak_groundTruth_br-peak_unwrap1_br)
                if peak_unwrap_python_br :
                    error_peak_gt_unwrap_python_br[row,col] =np.linalg.norm(peak_groundTruth_br-peak_unwrap_python_br)
                if peak_EATAN1_br :
                    error_peak_gt_EATAN1_br[row,col] =np.linalg.norm(peak_groundTruth_br-peak_EATAN1_br)
                if peak_EATAN2_br:
                    error_peak_gt_EATAN2_br[row,col] =np.linalg.norm(peak_groundTruth_br-peak_EATAN2_br)
                if peak_unlimited_br :
                    error_peak_gt_unlimited_br[row,col] =np.linalg.norm(peak_groundTruth_br-peak_unlimited_br)

                if peak_MDACM_hr :
                    error_peak_gt_MDACM_hr[row,col] = np.linalg.norm(peak_groundTruth_hr-peak_MDACM_hr)
                if peak_unwrap1_hr :
                    error_peak_gt_unwrap1_hr[row,col] = np.linalg.norm(peak_groundTruth_hr-peak_unwrap1_hr)
                if peak_unwrap_python_hr :
                    error_peak_gt_unwrap_python_hr[row,col] =np.linalg.norm(peak_groundTruth_hr-peak_unwrap_python_hr)
                if peak_EATAN1_hr :
                    error_peak_gt_EATAN1_hr[row,col] =np.linalg.norm(peak_groundTruth_hr-peak_EATAN1_hr)
                if peak_EATAN2_hr :
                    error_peak_gt_EATAN2_hr[row,col] =np.linalg.norm(peak_groundTruth_hr-peak_EATAN2_hr)
                if peak_unlimited_hr :
                    error_peak_gt_unlimited_hr[row,col] =np.linalg.norm(peak_groundTruth_hr-peak_unlimited_hr)

            print(row)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if MC_flag== True:
            plt.figure()
            plt.imshow((error_gt_MDACM+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_gt_MDACM_sin_"+bio_model+"_SNR" +str(snr_element)+"_rbm_"+rbm_type[k]+".pdf")
            #plt.title(' error_d_gt_MDACM')

            plt.figure()
            plt.imshow((error_gt_unwrap1+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title('error_d_gt_unwrap1')
            plt.savefig("error_gt_unwrap1_"+bio_model+"_SNR"+str(snr_element)+"_rbm_"+rbm_type[k]+".pdf")

            plt.figure()
            plt.imshow((error_gt_unwrap_python+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title('error_d_gt_unwrap_python')
            plt.savefig("error_gt_unwrap_python_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow((error_gt_EATAN1+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title('error_d_gt_EATAN1')
            plt.savefig("error_gt_EATAN1_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow((error_gt_EATAN2+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title('error_d_gt_EATAN2')
            plt.savefig("error_gt_EATAN2_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow((error_gt_unlimited+epsilon)[::-1,:], aspect="auto",extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(0, 10)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title('error_d_gt_unlimited')
            plt.savefig("error_gt_unlimited_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_MDACM_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            # plt.title(' error_peak_gt_MDACM_br(dB)')
            plt.savefig("error_peak_gt_MDACM_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")


            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unwrap1_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unwrap1_br(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unwrap1_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")


            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unwrap_python_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unwrap_python_br(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unwrap_python_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")



            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_EATAN1_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_EATAN1_br(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_EATAN1_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_EATAN2_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_EATAN2_br(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_EATAN2_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unlimited_br + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unlimited_br(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unlimited_br_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_MDACM_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title(' error_peak_gt_MDACM_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_MDACM_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unwrap1_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unwrap1_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unwrap1_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unwrap_python_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unwrap_python_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unwrap_python_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_EATAN1_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_EATAN1_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_EATAN1_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_EATAN2_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_EATAN2_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_EATAN2_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            plt.figure()
            plt.imshow(10 * np.log10(error_peak_gt_unlimited_hr + epsilon)[::-1,:], aspect="auto",
                       extent=[Fs_all[0], Fs_all[-1], Amp_range[0], Amp_range[-1], ])
            # plt.imshow(20*np.log10(mean_over_i)[:,::1],aspect="auto", extent=[Amp_range[0], Amp_range[-1], Frequency_range[0], Frequency_range[-1]])
            plt.colorbar()
            plt.clim(-100, 0)
            # plt.title('error_peak_gt_unlimited_hr(dB)')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('RBM Amplitude [m]')
            plt.savefig("error_peak_gt_unlimited_hr_"+bio_model+"_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")








        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot time domain signal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else :

            plt.figure(figsize=(8, 6))
            plt.plot(t,np.angle(expo_x_total), color=fun.custom_colors.get(0),label='wrapped signal')
            plt.plot(t,ground_truth_signal,color=fun.custom_colors.get(14), marker='+',label = 'Ground truth')
            plt.plot(t,MDACM_signal,color=fun.custom_colors.get(2),marker='o',label ='MDACM')
            # plt.plot(t,unwrap1_signal,color=fun.custom_colors.get(4), marker='p',label ='phase unwrap - realtime')
            plt.plot(t,phase_unwrap_python_signal,color=fun.custom_colors.get(3),marker ='^',label ='phase unwrap')
            plt.plot(t,EATAN1_signal,color=fun.custom_colors.get(9),marker='*',label ='EATAN1')
            plt.plot(t,EATAN2_signal, color=fun.custom_colors.get(5),marker ='v',label ='EATAN2',linewidth = 0.8, markersize = 5)
            plt.plot(t,phase_unlimited,color=fun.custom_colors.get(6),marker ='<',label='Unlimited Sampling',linewidth = 2)
            plt.xlabel('Time [s]', fontsize=fun.myfontsize)
            plt.ylabel('Phase [radian]', fontsize=fun.myfontsize)
            # plt.grid()
            plt.ylim([-30, 30])
            plt.legend(fontsize=fun.myfontsize-1)
            plt.savefig("UnwrapedPhaseMethods_d_" + bio_model + "_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")


            plt.figure(figsize=(8, 6))
            plt.plot(t,np.angle(expo_x_total), color=fun.custom_colors.get(0),label='wrapped signal')
            plt.plot(t,ground_truth_signal-np.mean(ground_truth_signal),color=fun.custom_colors.get(14), marker='+',label = 'Ground truth')
            plt.plot(t,MDACM_signal-np.mean(MDACM_signal),color=fun.custom_colors.get(2),marker='o',label ='MDACM')
            # plt.plot(t,unwrap1_signal-np.mean(unwrap1_signal),color=fun.custom_colors.get(4), marker='p',label ='phase unwrap - realtime')
            plt.plot(t,phase_unwrap_python_signal-np.mean(phase_unwrap_python_signal),color=fun.custom_colors.get(3),marker ='^',label ='phase unwrap')
            plt.plot(t,EATAN1_signal-np.mean(EATAN1_signal),color=fun.custom_colors.get(9),marker='*',label ='EATAN1')
            plt.plot(t,EATAN2_signal-np.mean(EATAN2_signal), color=fun.custom_colors.get(5),marker ='v',label ='EATAN2',linewidth = 0.8, markersize = 5)
            plt.plot(t,phase_unlimited-np.mean(phase_unlimited),color=fun.custom_colors.get(6),marker ='.',label='Unlimited Sampling',linewidth = 2)
            plt.xlabel('Time [s]', fontsize=fun.myfontsize)
            plt.ylabel('Phase [radian]', fontsize=fun.myfontsize)
            # plt.grid()
            plt.ylim([-30, 30])
            plt.legend(fontsize=fun.myfontsize-1)
            plt.savefig("UnwrapedPhaseMethods_d_meanOmited_" + bio_model + "_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot breathing interval Frequency domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # plt.figure(figsize=(4, 6))
            plt.subplot(1,2,1)
            if peak_groundTruth_br is not None:
                plt.plot(peak_groundTruth_br,10*np.log10(fft_windowed_signal_GroundTruth_breathing[interval_breathing_gt[filtered_peaks_groundTruth_br[best_peak_index_groundTruth_br]]]),color=fun.custom_colors.get(14), marker='o',markersize = 25)
            if (peak_MDACM_br) is not None:
                plt.plot(peak_MDACM_br,10*np.log10(fft_windowed_signal_MDACM_breathing[interval_breathing[filtered_peaks_MDACM_br[best_peak_index_MDACM_br]]]),color=fun.custom_colors.get(2), marker='*',markersize = 20)
            # if (peak_unwrap1_br) is not None:
            #     plt.plot(peak_unwrap1_br,10*np.log10(fft_windowed_signal_phase_unwrap1_breathing[interval_breathing[filtered_peaks_unwrap1_br[best_peak_index_unwrap1_br]]]),color=fun.custom_colors.get(4), marker='*',markersize = 20)
            if (peak_unwrap_python_br) is not None:
                plt.plot(peak_unwrap_python_br,10*np.log10(fft_windowed_signal_unwrap_python_breathing[interval_breathing[filtered_peaks_unwrap_python_br[best_peak_index_unwrap_python_br]]]),color=fun.custom_colors.get(3), marker='*',markersize = 20)
            if (peak_EATAN1_br) is not None:
                plt.plot(peak_EATAN1_br,10*np.log10(fft_windowed_signal_EATAN1_breathing[interval_breathing[filtered_peaks_EATAN1_br[best_peak_index_EATAN1_br]]]),color=fun.custom_colors.get(9), marker='*',markersize = 20)
            if (peak_EATAN2_br) is not None:
                plt.plot(peak_EATAN2_br,10*np.log10(fft_windowed_signal_EATAN2_breathing[interval_breathing[filtered_peaks_EATAN2_br[best_peak_index_EATAN2_br]]]),color=fun.custom_colors.get(5), marker='*',markersize = 20)
            if (peak_unlimited_br) is not None:
                plt.plot(peak_unlimited_br,10*np.log10(fft_windowed_signal_phase_unlimited_breathing[interval_breathing[filtered_peaks_unlimited_br[best_peak_index_unlimited_br]]]),color=fun.custom_colors.get(6), marker='*',markersize = 20)


            plt.plot(f_axis,10*np.log10(fft_windowed_signal_GroundTruth_breathing),color=fun.custom_colors.get(14), marker='+',label = 'Ground truth')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_MDACM_breathing),color=fun.custom_colors.get(2),marker='_',label ='MDACM')
            # plt.plot(f_axis,10*np.log10(fft_windowed_signal_phase_unwrap1_breathing),color=fun.custom_colors.get(4), marker='p',label ='phase unwrap - realtime')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_unwrap_python_breathing),color=fun.custom_colors.get(3),marker ='^',label ='phase unwrap')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_EATAN1_breathing),color=fun.custom_colors.get(9),marker='d',label ='EATAN1')
            plt.plot(f_axis,np.transpose(10*np.log10(fft_windowed_signal_EATAN2_breathing)), color=fun.custom_colors.get(5),marker ='v',label ='EATAN2',linewidth = 0.8, markersize = 5)
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_phase_unlimited_breathing),color=fun.custom_colors.get(6),marker ='.',label='Unlimited Sampling',linewidth = 2)

            plt.xlabel('Frequency [Hz]', fontsize=fun.myfontsize)
            plt.ylabel('FFT Amplitude [dB]', fontsize=fun.myfontsize)
            plt.xlim([low_breathing,high_breathing])
            plt.axvspan(low_breathing, high_breathing, color='yellow', alpha=0.3)
            plt.tight_layout()
            # plt.grid()
            plt.legend(fontsize=fun.myfontsize-1)
            plt.savefig("UnwrapedPhaseMethods_FFT_breathing_" + bio_model + "_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot Heart Rate interval Frequency domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #plt.figure(figsize=(4, 6))
            plt.subplot(1, 2, 2)
            if peak_groundTruth_hr is not None:
                plt.plot(peak_groundTruth_hr,10*np.log10(fft_windowed_signal_GroundTruth_heart[interval_heart_gt[filtered_peaks_groundTruth_hr[best_peak_index_groundTruth_hr]]]),color=fun.custom_colors.get(14), marker='o',markersize = 25)
            if (peak_MDACM_hr) is not None:
                plt.plot(peak_MDACM_hr,10*np.log10(fft_windowed_signal_MDACM_heart[interval_heart[filtered_peaks_MDACM_hr[best_peak_index_MDACM_hr]]]),color=fun.custom_colors.get(2), marker='*',markersize = 20)
            # if (peak_unwrap1_hr) is not None:
            #     plt.plot(peak_unwrap1_hr,10*np.log10(fft_windowed_signal_phase_unwrap1_heart[interval_heart[filtered_peaks_unwrap1_hr[best_peak_index_unwrap1_hr]]]),color=fun.custom_colors.get(4), marker='*',markersize = 20)
            if (peak_unwrap_python_hr) is not None:
                plt.plot(peak_unwrap_python_hr,10*np.log10(fft_windowed_signal_unwrap_python_heart[interval_heart[filtered_peaks_unwrap_python_hr[best_peak_index_unwrap_python_hr]]]),color=fun.custom_colors.get(3), marker='*',markersize = 20)
            if (peak_EATAN1_hr) is not None:
                plt.plot(peak_EATAN1_hr,10*np.log10(fft_windowed_signal_EATAN1_heart[interval_heart[filtered_peaks_EATAN1_hr[best_peak_index_EATAN1_hr]]]),color=fun.custom_colors.get(9), marker='*',markersize = 20)
            if (peak_EATAN2_hr) is not None:
                plt.plot(peak_EATAN2_hr,10*np.log10(fft_windowed_signal_EATAN2_heart[interval_heart[filtered_peaks_EATAN2_hr[best_peak_index_EATAN2_hr]]]),color=fun.custom_colors.get(5), marker='*',markersize = 20)
            if (peak_unlimited_hr) is not None:
                plt.plot(peak_unlimited_hr,10*np.log10(fft_windowed_signal_phase_unlimited_heart[interval_heart[filtered_peaks_unlimited_hr[best_peak_index_unlimited_hr]]]),color=fun.custom_colors.get(6), marker='*',markersize = 20)


            plt.plot(f_axis,10*np.log10(fft_windowed_signal_GroundTruth_heart),color=fun.custom_colors.get(14), marker='+',label = 'Ground truth')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_MDACM_heart),color=fun.custom_colors.get(2),marker='_',label ='MDACM')
            # plt.plot(f_axis,10*np.log10(fft_windowed_signal_phase_unwrap1_heart),color=fun.custom_colors.get(4), marker='p',label ='phase unwrap - realtime')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_unwrap_python_heart),color=fun.custom_colors.get(3),marker ='^',label ='phase unwrap')
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_EATAN1_heart),color=fun.custom_colors.get(9),marker='d',label ='EATAN1')
            plt.plot(f_axis,np.transpose(10*np.log10(fft_windowed_signal_EATAN2_heart)), color=fun.custom_colors.get(5),marker ='v',label ='EATAN2',linewidth = 0.8, markersize = 3)
            plt.plot(f_axis,10*np.log10(fft_windowed_signal_phase_unlimited_heart),color=fun.custom_colors.get(6),marker ='.',label='Unlimited Sampling',linewidth = 1.5)

            plt.xlabel('Frequency [Hz]', fontsize=fun.myfontsize)
            plt.ylabel('FFT Amplitude [dB]', fontsize=fun.myfontsize)
            plt.xlim([low_heart, high_heart])
            plt.axvspan(low_heart, high_heart, color='red', alpha=0.3)
            plt.tight_layout()
            # plt.grid()
            plt.legend(fontsize=fun.myfontsize-1)
            plt.savefig("UnwrapedPhaseMethods_FFT_heart_" + bio_model + "_SNR" + str(snr_element) + "_rbm_" + rbm_type[k] + ".pdf")
            # plt.show()
