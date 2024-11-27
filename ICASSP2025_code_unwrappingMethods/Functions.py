import math
import numpy as np
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy

from scipy.fftpack import fft, ifft ,fftfreq
from scipy.fftpack import fftshift, ifftshift

from scipy.signal import find_peaks
from scipy import signal

from scipy.signal import butter, lfilter, freqz
from scipy .signal import firwin, remez,kaiser_atten, kaiser_beta
import scipy.signal as sig


from scipy.signal import resample

myfontsize = 14
custom_colors = {
    1: [0, 0.4470, 0.7410],
    2: [0.8500, 0.3250, 0.0980],
    3: [0.9290, 0.6940, 0.1250],
    4: [0.4940, 0.1840, 0.5560],
    5: [0.4660, 0.6740, 0.1880],
    6: [0.3010, 0.7450, 0.9330],
    7: [0.6350, 0.0780, 0.1840],
    8: [1, 0, 0],   # red
    9: [0, 1, 0],   # green
    10: [0, 0, 1],  # blue
    11: [0, 1, 1],  # cyan
    12: [1, 0, 1],  # magenta
    13: [1, 1, 0],  # yellow
    14: [0, 0, 0]   # black
    # Add more colors and their associated numbers here
}








################# Random Body Movement ####################################

def random_body_generation(t,dt,StopTime, rbm_duration,rbm_amp, rbm_type, F_rbm, phi=np.pi/3, BW=4,is_rbm= 'Yes'):
    '''

    :param t: sec
    :param dt: 1/Fs
    :param StopTime: 20 sec
    :param rbm_duration: parial duration of the whole time [sec]
    :param rbm_amp: m
    :param rbm_type: sin/ramp/general/comb_general
    :param F_rbm: frequency
    :param phi: radian
    :param BW: bandwidth
    :param is_rbm: yes/no
    :return: signal of random body movement
    '''

    if is_rbm == 'Yes':
        t_rbm_duration = np.arange(np.round(rbm_duration / dt)) * dt
        rbm_sig = np.zeros(t.shape)
        I_start = int(np.round((StopTime-rbm_duration)/2/dt))
        x_rbm =  np.zeros(t_rbm_duration.shape)
        # t_start_rbm = F_rbm*self.t*9/self.StopTime
        # t_duration_rbm = F_rbm*self.t*2/self.StopTime

        if rbm_type == 'sin':
            x_rbm = rbm_amp*np.sin(2*np.pi*F_rbm*t_rbm_duration + phi)


        elif rbm_type == 'ramp':
            x_rbm = rbm_amp*(t_rbm_duration/rbm_duration)

        elif rbm_type == 'general':
            random_phase = np.random.rand(len(t_rbm_duration), 1)
            x_rbm_omega = np.exp(1j*2*np.pi*random_phase)
            z = np.zeros(len(x_rbm_omega))
            z[1:np.round(BW*rbm_duration)] = 1
            x_rbm1 = z*np.squeeze(x_rbm_omega)
            x_rbm2 = np.fft.ifft(x_rbm1)
            x_rbm = x_rbm2.real
            abs_x_rbm = np.abs(x_rbm)
            x_rbm = rbm_amp*x_rbm/abs_x_rbm.max()


        elif rbm_type == 'comb_general':

            BRstart = 0.2
            BRend = 0.7
            HRstart = 0.8
            HRend = 4
            cr = rbm_duration

            if rbm_duration!=0 or rbm_amp!=0:
                random_phase = np.random.rand(len(t_rbm_duration), 1)
                x_rbm_omega = np.exp(1j*2*np.pi*random_phase)
                z = np.zeros(len(x_rbm_omega))
                z[0:int(BRstart*cr)] = 1
                z[int(BRend*cr):int(HRstart*cr)] = 1
                z[int(HRend*cr):int(BW*cr)] = 1

                x_rbm1 = z*np.squeeze(x_rbm_omega)
                x_rbm2 = np.fft.ifft(x_rbm1)
                x_rbm = x_rbm2.real
                abs_x_rbm = np.abs(x_rbm)
                x_rbm = rbm_amp*x_rbm/abs_x_rbm.max()

        rbm_sig[I_start:I_start+len(x_rbm)] = x_rbm
        rbm_sig[:I_start] = x_rbm[0]
        rbm_sig[I_start+len(x_rbm):] = x_rbm[-1]
    else:
        rbm_sig = 0

    return rbm_sig


def BR_math_model(Pm, T, coeff, fbr, Fs, t):
    '''

    :param Pm: Breathing rate amplitude [m]
    :param T: StopTime
    :param coeff: inspiratory-expiratory time ratio.
    :param fbr:breathing rate frequency
    :param Fs: Sampling Frequency
    :param t: stepwise duration
    :return: signal of breathing rate
    '''
    T_each_period = 1 / fbr
    coeff = 1  # inspiratory to expiratory coefficient
    Te = (T_each_period) / (1 + coeff)
    Ti = coeff * Te
    Ti_coeff = coeff / (1 + coeff)
    tau = 0.2 * Te

    # t=np.linspace(0,T, int(T*Fs))

    tprime = t % T_each_period
    # print(tprime)

    signal_br = np.zeros(tprime.shape)
    signal_br[tprime <= Ti] = (-Pm / (Ti * Te)) * pow(tprime[tprime <= Ti], 2) + (Pm * T_each_period / (Ti * Te)) * \
                              tprime[tprime <= Ti]
    signal_br[tprime > Ti] = (Pm / (1 - np.exp(-Te / tau))) * (
                np.exp(-(tprime[tprime > Ti] - Ti) / tau) - np.exp(-Te / tau))

    return signal_br, t


def HR_math_model(alpha, T, fhr, Fs,t):  # fbr is beat per minutes Fs= Sampling rate in Hz, alpha is the amplitude of HR signal.
    # t=np.linspace(0,T, int(T*Fs))
    number_of_periods_hr = T * fhr

    HRpulse = np.zeros(t.shape)
    for i in range(int(np.ceil(number_of_periods_hr) + 1)):
        T_pulse = (i - 1) / fhr
        HRpulse += alpha * np.exp(-1 * (t - T_pulse) ** 2 / 0.0015)

    return HRpulse, t


def sig_generator(t,Amp_br, Fc):
    x = Amp_br * np.cos(2 * np.pi * Fc * t)
    return x


def get_total_signal(x_rbm, t, Amp_br, Amp_hr, Fc_br, Fc_hr, Fs, StopTime, coeff, is_rbm, bio_model):
    x_total = 0
    if bio_model == 'sin':
        x_br = sig_generator(t, Amp_br, Fc_br)
        x_hr = sig_generator(t, Amp_hr, Fc_hr)

    elif bio_model == 'Albanese':
        x_br, t1 = BR_math_model(Amp_br, StopTime, coeff, Fc_br, Fs, t)
        x_hr, t2 = HR_math_model(Amp_hr, StopTime, Fc_hr, Fs, t)

    if is_rbm == 'Yes':
        x_total = x_br + x_hr + x_rbm
    elif is_rbm == 'No':
        x_total = x_br + x_hr

    x_without_rbm = x_br + x_hr

    return x_total, x_without_rbm








################################ Phase Unwrapping Procedure ###########################################


def phase_unwrap_python(expo_x_total):
    p = np.angle(expo_x_total)
    dd = np.diff(p)
    ddmod = np.mod(dd + np.pi, 2 * np.pi) - np.pi
    ph_correct = ddmod - dd
    return p + np.hstack((0, np.cumsum(ph_correct)))

def phase_unwrap1(expo_x_total):
    xw = np.angle(expo_x_total)
    xu = xw

    for i in range(1,len(xw)):
        difference = xw[i]-xw[i-1]

        if difference > np.pi:
            xu[i:] = xu[i:] - 2*np.pi
        elif difference < -np.pi:
            xu[i:] = xu[i:] + 2*np.pi

    return xu


def phase_unwrap1_RealTime(expo_x_total):
    xw = np.angle(expo_x_total)
    xu = np.zeros(len(xw))
    ccs = 0
    for i in range(1,len(xw)):
        difference = xw[i]-xw[i-1]
        if difference > np.pi:
            ccs -= 2*np.pi
        elif difference < -np.pi:
            ccs += 2*np.pi
        xu[i] = xw[i] + ccs
    return xu

''' MDACM '''
def MDACM(expo_x_total):
    I = np.real(expo_x_total)
    I_diff = np.expand_dims(np.diff(I), axis=0)
    I = np.expand_dims(I, axis=0)

    Q = np.imag(expo_x_total)
    Q_diff = np.expand_dims(np.diff(Q), axis=1)
    Q = np.expand_dims(Q, axis=1)

    temp = np.multiply(I[0,1:], np.transpose(Q_diff)) - np.multiply(I_diff, np.transpose(Q[1:,0]))

    DACM_phase = np.hstack((0, np.cumsum(np.reshape((temp),[len(Q_diff),1]))))
    return DACM_phase

''' 2nd Ord MDACM'''
def sec_ord_MDACM(expo_x_total):
    I = np.real(expo_x_total)
    I_diff_1 = np.expand_dims(np.diff(I), axis=0)
    I_diff_2 = np.expand_dims(np.diff(I,2), axis=0)
    I = np.expand_dims(I, axis=0)

    Q = np.imag(expo_x_total)
    Q_diff_1 = np.expand_dims(np.diff(Q), axis=1)
    Q_diff_2 = np.expand_dims(np.diff(Q,2), axis=1)
    Q = np.expand_dims(Q, axis=1)

    temp = (np.multiply(I[0,2:],np.transpose(Q_diff_2)) - np.multiply(I_diff_2, np.transpose(Q[2:,0])))
    DACM_phase = np.hstack((0, np.cumsum(np.reshape((temp),[len(Q_diff_2),1]))))

    MDACM_2_ord = cumsum_func(DACM_phase)
    MDACM_2_ord = MDACM_2_ord - np.mean(MDACM_2_ord)
    return MDACM_2_ord


''' 2nd Ord DACM'''
def sec_ord_DACM(expo_x_total):
    I = np.real(expo_x_total)
    I_diff_1 = np.expand_dims(np.diff(I), axis=0)
    I_diff_2 = np.expand_dims(np.diff(I,2), axis=0)
    I = np.expand_dims(I, axis=0)

    Q = np.imag(expo_x_total)
    Q_diff_1 = np.expand_dims(np.diff(Q), axis=1)
    Q_diff_2 = np.expand_dims(np.diff(Q,2), axis=1)
    Q = np.expand_dims(Q, axis=1)

    num_derivative   = np.multiply(I[0,2:],np.transpose(Q_diff_2)) - np.multiply(I_diff_2, np.transpose(Q[2:,0]))
    denum            = np.power(I[0,2:],2)+np.power(np.transpose(Q[2:,0]),2)
    num              = (np.multiply(I[0,2:], np.transpose(Q_diff_1[1:,0])) - np.multiply(I_diff_1[0,1:], np.transpose(Q[2:,0])))
    denum_derivative = np.multiply(I[0,2:], I_diff_1[0,1:]) - np.multiply(Q_diff_1[1:,0], Q[2:,0])
    denum_power = np.power((np.power(I[0,2:],2)+np.power(np.transpose(Q[2:,0]),2)),2)

    temp = (num_derivative*denum - num*denum_derivative)/denum_power

    DACM_phase = np.hstack((0, np.cumsum(np.reshape((temp),[len(Q_diff_2),1]))))

    DACM_2_ord = cumsum_func(DACM_phase)
    DACM_2_ord = DACM_2_ord - np.mean(DACM_2_ord)

    return DACM_2_ord



'''EATAN 1'''

def phase_compensation(phase):
    len_phase = len(phase)
    diff_p = np.diff(phase)
    diff_num = diff_p / (2 * np.pi)
    round_down = 1*(np.abs(np.mod(diff_num, 1)) <= 0.5)
    diff_num[round_down] = np.fix(diff_num[round_down])
    diff_num = np.round(diff_num)
    diff_num[np.abs(diff_p) < np.pi] = 0
    phase[1:len_phase] -= (2 * np.pi) * np.cumsum(diff_num, axis=0)
    return phase



def EATAN1(expo_x_total):
    phase_atan2 = np.angle(expo_x_total)
    phase_dc = phase_compensation(np.diff(phase_atan2))
    phase_EATAN1 = np.concatenate(([phase_atan2[0]], phase_dc), axis=0)
    return np.cumsum(phase_EATAN1)

# phase_EATAN1 = EATAN1(expo_x_total)
# plt.figure()
# plt.plot(phase_EATAN1)
# plt.show()
''' EATAN 2 version 2'''


''' EATAN 2 '''
# def k_finding(expo_x_total):
#     phase_atan2 = np.angle(expo_x_total)
#     d_com = np.diff(phase_atan2)
#     k = 0
#     while np.max(d_com) > np.pi:
#         k +=1
#         d_com = np.diff(phase_atan2,k)

def find_k_cheatingMethod(signal, lambda_val,Max_N_K):
    k = -1
    while True:
        k = k + 1
        # print(k)

        
        #if (np.max(2         / lambda_val * np.diff(signal, k)) < np.pi) or k > Max_N_K:
        #Mohammads version
        if (np.max(4 * np.pi / lambda_val * np.diff(signal, k)) < np.pi) or k > Max_N_K:

            k = Max_N_K
            break

    if k < 2:
        k = 2
    # print(k)
    return k




def EATAN2(expo_x_total,ground_truth_signal,t,lambda_val,Max_N_K):
    # ts = StopTime

    angle = np.angle(expo_x_total)
    k = find_k_cheatingMethod(ground_truth_signal, lambda_val,Max_N_K)
    #print('k is', k)
    #pEATAN = np.zeros([1, int(Fs * ts)])
    pEATAN = np.zeros([1,len(angle)])
    phase_dc = phase_compensation(np.diff(angle, k - 1))

    pEATAN[0, k - 1:] = phase_dc[:len(pEATAN[0, k - 1:])]
    # print(pEATAN.shape)
    # t = np.arange(1 / Fs, ts + (1 / Fs), 1 / Fs)

    for index in range(k, 1, -1):
        # print('index',int(index))
        pEATAN[0, index - 1:] = pEATAN[0, index - 1:] - np.mean(pEATAN[0, index - 1:])
        pEATAN[0, index - 2:] = np.cumsum(pEATAN[0, index - 2:])
        # print('t',t[index-2:].shape)
        # print('pEATAN' ,len(pEATAN[0,index-2:]))
        corIndex = len(t[index - 2:]) - len(pEATAN[0, index - 2:])
        # print('corIndex', corIndex)
        # print('reduced t', t[index-2:-corIndex].shape)
        # print(t[index-2:len(t[index-2:])-corIndex].shape)
        # print('total t',len(t))
        # print('total peatan', len(pEATAN[0,:]))
        tt = np.polyfit(t[index - 2:len(t) - corIndex], pEATAN[0, index - 2:], k - index + 1)
        pEATAN[0, index - 2:] = pEATAN[0, index - 2:] - np.polyval(tt, t[index - 2:len(t) - corIndex])

    pEATAN = pEATAN - np.mean(pEATAN)
    # print(pEATAN.shape)
    tt = np.polyfit(t[0:len(t) - corIndex], pEATAN[0, :], k)
    phase = pEATAN - np.polyval(tt, t[:len(pEATAN[0, :])])

    return phase[0,:]



''' Unlimited Sampling'''

def cumsum_func(s):
    zero_array = np.zeros([1])
    s = np.concatenate((zero_array,s))
    s = np.cumsum(s)
    return s


def Unlimited_Sampling(expo_x_total,Fs,Br,lambda_val,Max_N_K):
    y = np.angle(expo_x_total)

    #Br = math.ceil(np.max(abs(ground_truth_signal))/(2*lambda_val))*2*np.pi

    BW = 0.5
    N =  np.ceil((np.log(np.pi)-np.log(Br))/np.log((1/Fs)*2*np.pi*BW*np.exp(1)))
    #N = 6
    print('N is', N)
    if N<1:
        N = 1
    # elif N >Max_N_K:
    #     N =Max_N_K

    dy = np.diff(y,int(N))
    T = np.pi
    mod_dy = np.mod(dy+T,2*T) -T
    s0 = mod_dy - dy
    J = np.min([6 * Br / np.pi, len(s0) - 1])

    s = s0
    for n in range(int(N)-1):
        s = cumsum_func(s)
        # s = 2*np.pi*np.ceil(np.floor(s/np.pi)/2)
        s_integrated = cumsum_func(s) #np.hstack((0, np.cumsum(s)))
        kn = np.floor((s_integrated[0]-s_integrated[int(J)])/(12*Br)+0.5)
        s = s + 2*np.pi*kn

    phase_unlimited = y + cumsum_func(s)
    return phase_unlimited


''' Peak Detection '''

def mypeaks(fft_windowed_signal,interval,distance_peaks,N_fft,Fs,f_axis):
    breathing_signal_region = fft_windowed_signal[interval]
    peaks, _ = find_peaks(breathing_signal_region, distance=int(max(1,distance_peaks*N_fft/Fs)))
    filtered_peaks = peaks[(peaks > 0) & (peaks < len(breathing_signal_region) - 1)]
    selected_peak = []
    best_peak_index = []
    if len(filtered_peaks) > 0:
        best_peak_index = np.argmax(breathing_signal_region[filtered_peaks])
        selected_peak = f_axis[interval[filtered_peaks[best_peak_index]]]

        # T(selected_peak)
    return selected_peak, best_peak_index, filtered_peaks


