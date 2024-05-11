#bibliotecas necessarias
import csv
import numpy as np
from scipy import signal

#funcoes uteis
from re import search
from os import get_terminal_size
from time import sleep
from copy import deepcopy
from sklearn.preprocessing import minmax_scale
from pylsl import StreamInfo, StreamOutlet, resolve_stream, StreamInlet
from concurrent.futures import ThreadPoolExecutor

#debug
# import matplotlib.pyplot as plt

class Eletroencefalograma:
    def __init__(self, freq, electrodes):
        self.electrodes = electrodes
        self.freq = freq
        self.data = None
    #end init/open
        

    def processSample(self, electrodes=[], sample=None, notch=0, lowcut=0, highcut=0):
        data = np.array(sample)

        #dominio do tempo
        data = data.swapaxes(1, 0)

        #remove colunas que nao serao usadas
        if electrodes:
            delete = [i for i in range(self.electrodes) if i not in electrodes]
            data = np.delete(data, delete, 0)

        
        # Aplica esses filtros lá no buffer size processing
        
        #filtros
     
        # if notch:
        #     # aplica o filtro notch no valor determinado e nos harmonicos
        #     while notch <= (self.freq/2):
        #         data = self.__butterNotch(data, notch)
        #         notch = notch*2

        # if lowcut:
        #     #Remove o que estiver abaixo de Lowcut
        #     data = self.__butterHighpass(data, lowcut)

        # if highcut:
        #     #Remove o que estiver acima de Highcut
        #     data = self.__butterLowpass(data, highcut)
        if(self.data is None):
            self.data = data
        else:
            self.data = np.concatenate((self.data, data), axis=1)
    #end configure


    def execute(self, output, bufferSize, refresh, scale=0, start=0, simulate=False, stream=False):
        seconds = start

        info = StreamInfo('Processed Data', 'Markers', 1, 0, 'float32', 'myuidw43536')
        outlet = StreamOutlet(info)

        print("looking for a OpenBCI EEG stream...")
        streams = resolve_stream('type', 'EEG')
        inlet = StreamInlet(streams[0])

        print("EEG Stream Found...")
        inlet.pull_sample()
        # After waiting for a 4 seconds initial buffer, pull the chunk from the LSL stream
        sleep(4.2)
        chunk = inlet.pull_chunk()
        eeg.processSample(electrodes=[1,2,3,4,5,6,7,8], sample=list(chunk[0]), notch=60, lowcut=5, highcut=35)

        print(chunk[0].__len__())

        with open(output + '.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Janela', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'])
            while True:
                #janela/buffer
                # if seconds < (start+bufferSize):
                if seconds < bufferSize:
                    continue
                # 1-2-3s x 256Hz + 4s x 256Hz (bufferSize)
                end = int((seconds * self.freq) + (bufferSize * self.freq))
                # 1-2-3s x 256Hz
                begin = int(seconds * self.freq)
                sleep(refresh)

                chunk = inlet.pull_chunk()
                eeg.processSample(electrodes=[1,2,3,4,5,6,7,8], sample=chunk[0], notch=0, lowcut=0, highcut=0)
                
                print(chunk[0].__len__())
                #welch
                f, psdWelch = signal.welch(self.data[:,begin:end])
                psdWelch = np.average(psdWelch, axis=0)
                features = list()
                for mi, ma in [(0, 4),(4, 8),(8, 12),(12, 30),(30, 100)]:
                    features.append(psdWelch[mi:ma])
                features = [np.average(f) for f in features]

                if scale:
                    features = minmax_scale(features, feature_range=(0, scale))

                #escreve no csv
                writer.writerow(np.insert(features, 0, seconds))

                # plot
                if simulate:
                    self.__consolePlot(bufferSize, seconds, features)
                
                # Stream data to the network
                if stream:
                    print(features)
                    # outlet.push_sample([int(features)])
                
                seconds += refresh
                
            #end while
        #end csv
    #end execute

    # ===============================================================================

    def __open(self, filename):
        data = list()
        with open(filename) as file:
            linhas = file.readlines()
        for linha in linhas:
            res = search('^\d{1,3},(?P<dado>(\ -?.+?,){%d})'%self.electrodes, linha)                
            if res:
                cols = res.group(1)
                data.append([float(d[1:]) for d in cols.split(',') if d])
        return np.array(data[1:])
    #end open
    

    def __consolePlot(self, bufferSize, second, features):
        terminal = get_terminal_size()

        r = int(terminal.columns/2) #range
        feat = minmax_scale(features, feature_range=(0, r))
        delta = '[' + ('=' * int(feat[0])) + (' ' * (r-int(feat[0]))) + ']'
        theta = '[' + ('=' * int(feat[1])) + (' ' * (r-int(feat[1]))) + ']'
        alpha = '[' + ('=' * int(feat[2])) + (' ' * (r-int(feat[2]))) + ']'
        beta = '[' + ('=' * int(feat[3])) + (' ' * (r-int(feat[3]))) + ']'
        gamma = '[' + ('=' * int(feat[4])) + (' ' * (r-int(feat[4]))) + ']'

        #minutos xd
        m = int(second / 60)
        m = m if m > 10 else f'0{m}'
        s = second % 60
        s = f'{s:.2f}' if s > 10 else f'0{s:.2f}'

        for i in range(terminal.lines):
            clear = '\x1b[F' + (' ' * terminal.columns) + '\x1b[A'
            print(clear)
        # print(f'Arquivo: {self.filename}')
        print(f'Simulação / Buffer {bufferSize}s / Segundo {second:.2f} ({m}:{s}):')
        print(f'  DELTA:\t{delta}\n  THETA:\t{theta}\n  ALPHA:\t{alpha}\n  BETA: \t{beta}\n  GAMMA:\t{gamma}')
    #end consoleplot

    # ===============================================================================

    #filters
    def __butterNotch(self, data, cutoff, var=1, order=4):
        nyq = self.freq * 0.5
        low = (cutoff - var) / nyq
        high = (cutoff + var) / nyq
        b, a = signal.iirfilter(order, [low, high], btype='bandstop', ftype="butter")
        return signal.filtfilt(b, a, data)

    def __butterLowpass(self, data, lowcut, order=4):
        nyq = self.freq * 0.5
        low = lowcut / nyq
        b, a = signal.butter(order, low, btype='lowpass')
        return signal.filtfilt(b, a, data)

    def __butterHighpass(self, data, highcut, order=4):
        nyq = self.freq * 0.5
        high = highcut / nyq
        b, a = signal.butter(order, high, btype='highpass')
        return signal.filtfilt(b, a, data)
    
    # def __butterBandpass(self, lowcut, highcut, order=4):
    #     nyq = self.freq * 0.5
    #     low = lowcut / nyq
    #     high = highcut / nyq
    #     b, a = signal.butter(order, [low, high], btype='bandpass')
    #     return signal.filtfilt(b, a, self.data)
    #end filters

    # ===============================================================================

    # #debug
    # def welchPlot(self, data):
    #     if len(data.shape) == 1:
    #         plt.plot(data[:55])
    #     else:
    #         for ch in range(data.shape[0]):
    #             plt.plot(data[ch,:55])
    #     plt.axvline(x=4, linestyle='--', color='red')
    #     plt.axvline(x=8, linestyle='--', color='blue')
    #     plt.axvline(x=12, linestyle='--', color='orange')
    #     plt.axvline(x=30, linestyle='--', color='purple')
    #     plt.show()
    # #end myplot

    # def matplotGraphs(self):
    #     for i in range(self.data.shape[0]):
    #         plt.plot(self.data[i,:])
    #     plt.title('Domínio do tempo')
    #     plt.show()

    #     for i in range(self.data.shape[0]):
    #         plt.psd(self.data[i,:], Fs=self.freq)
    #     plt.title('Domínio da frequência')
    #     plt.show()
        
    #     for i in range(self.data.shape[0]):
    #         plt.specgram(self.data[i,:], Fs=self.freq)
    #     plt.title('Espectrograma')
    #     plt.show()
    # #end matplotgraphs
#end class eeg

# =========================================================

if __name__ == "__main__":
    eeg = Eletroencefalograma(256, 8)

    # After the initial buffer, start a ThreadPoolExecutor to process them and re-stream to the network

    # Scale is height of screen minus object height

    eeg.execute(output="teste", bufferSize=4, refresh=1, scale=1030, start=6, simulate=False, stream=True)
    # eeg.matplotGraphs()

