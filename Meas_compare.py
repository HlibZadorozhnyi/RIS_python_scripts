import matplotlib.pyplot as plt
import skrf as rf
import numpy as np

def main():

    rf.stylely()

    show_voltages = [0.01, 19.8]

    voltages    = [ 0.01,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19, 19.8 ]
    corr_meas1 = [  240, 240, 240, 240, 240, 240, 240, 240, 240, 600, 600, 600, 240, 600, 600, 600, 240, 600, 600, 240,  600 ]
    corr_meas2 = [  240, 240, 240, 240, 240, 240, 240, 240, 240, 600, 600, 600, 600, 600, 960, 600, 600, 600, 600, 960,  600 ]
    corr_meas3 = [  240, 240, 240, 240, 240, 240, 240, 240, 240, 960, 600, 600, 600, 600, 960, 600, 600, 600, 600, 600,  600 ]
    corr_simUC = [  160, 160, 160, 160, 160, 160, 160, 160, 160, 160, -1*200, -1*200, -1*200, -1*200, -1*200, -1*200, -1*200, -1*200, -1*200, -1*200,  -1*200 ]

    corr_meas1 = np.asarray(corr_meas1) - 20
    corr_meas2 = np.asarray(corr_meas2) - 20
    corr_meas3 = np.asarray(corr_meas3) - 20
    corr_simUC = -1*np.asarray(corr_simUC)

    setup1 = MeasurementSetup()
    setup1.owner       = 'Kyiv'            # owner of the S-parameters: Kyiv or Braunschweig
    setup1.date        = '2025_01_16'      # Date of the measurement
    setup1.ports       = '1'               # 1 or 2 port setup
    setup1.spacing     = 'None'             # spacing of the 2 port setup ['Hor', 'Ver']
    setup1.antenna     = 'ant3'            # antenna for the measurement ['ant1', 'ant2', 'ant3', 'openWR90']
    setup1.distance    = '3300'            # distance of the measurement in [mm]
    setup1.file        = 's1p'             # depends on the measurement system: s1p or s2p
    setup1.param       = 's11'             # select the Sparam to apply the Time Analyzis
    setup1.postfix     = ''                # optional for Braunschweig ['_1', '_2', ..., '_7', '_8']
    setup1.getPath()
    setup1.getlabel(voltages)

    setup2 = MeasurementSetup()
    setup2.owner       = 'Kyiv'            # owner of the S-parameters: Kyiv or Braunschweig
    setup2.date        = '2025_01_16'      # Date of the measurement
    setup2.ports       = '2'               # 1 or 2 port setup
    setup2.spacing     = 'Hor'             # spacing of the 2 port setup ['Hor', 'Ver']
    setup2.antenna     = 'ant3'            # antenna for the measurement ['ant1', 'ant2', 'ant3', 'openWR90']
    setup2.distance    = '3300'            # distance of the measurement in [mm]
    setup2.file        = 's2p'             # depends on the measurement system: s1p or s2p
    setup2.param       = 's21'             # select the Sparam to apply the Time Analyzis
    setup2.postfix     = ''                # optional for Braunschweig ['_1', '_2', ..., '_7', '_8']
    setup2.getPath()
    setup2.getlabel(voltages)

    setup3 = MeasurementSetup()
    setup3.owner       = 'Kyiv'            # owner of the S-parameters: Kyiv or Braunschweig
    setup3.date        = '2025_01_16'      # Date of the measurement
    setup3.ports       = '2'               # 1 or 2 port setup
    setup3.spacing     = 'Ver'             # spacing of the 2 port setup ['Hor', 'Ver']
    setup3.antenna     = 'ant3'            # antenna for the measurement ['ant1', 'ant2', 'ant3', 'openWR90']
    setup3.distance    = '3300'            # distance of the measurement in [mm]
    setup3.file        = 's2p'             # depends on the measurement system: s1p or s2p
    setup3.param       = 's21'             # select the Sparam to apply the Time Analyzis
    setup3.postfix     = ''                # optional for Braunschweig ['_1', '_2', ..., '_7', '_8']
    setup3.getPath()
    setup3.getlabel(voltages) 
    
    gate = TimeGatingSettings()
    gate.start = 22
    gate.stop = 25
    gate.unit = 'ns'
    gate.mode = 'bandpass'
    gate.window = ('kaiser',5) # 'boxcar', 'hann' or 'hamming'
    gate.method = 'fft'

    meas1 = MeasurementResult(setup1, gate, voltages)
    meas2 = MeasurementResult(setup2, gate, voltages)
    meas3 = MeasurementResult(setup3, gate, voltages)

    simUC = SimulationResult(voltages)

    meas1.setCorrection(corr_meas1)
    meas2.setCorrection(corr_meas2)
    meas3.setCorrection(corr_meas3)
    simUC.setCorrection(corr_simUC)

    plt.figure(1)
    for i in range(len(show_voltages)):
        diff = np.asarray(voltages) - show_voltages[i]
        index = np.argmin(np.abs(diff))
        plt.plot(meas1.freq, meas1.phase_corr[index], label=setup1.label[index])
        plt.plot(meas2.freq, meas2.phase_corr[index], label=setup2.label[index])
        plt.plot(meas3.freq, meas3.phase_corr[index], label=setup3.label[index])
        plt.plot(simUC.freq, simUC.phase_corr[index], label="Simulated UC, V={0}".format(voltages[index]), ls='--', lw=3)
        plt.legend()
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Phase [deg]")
        plt.title("Phase of the RIS reflection R")
        plt.xlim([8, 15])
        plt.ylim([-400, 100])
    
    plt.figure(2)
    for i in range(len(show_voltages)):
        diff = np.asarray(voltages) - show_voltages[i]
        index = np.argmin(np.abs(diff))
        plt.plot(meas1.freq, np.abs(meas1.R_origin[index]), label=setup1.label[index])
        plt.plot(meas2.freq, np.abs(meas2.R_origin[index]), label=setup2.label[index])
        plt.plot(meas3.freq, np.abs(meas3.R_origin[index]), label=setup3.label[index])
        plt.plot(simUC.freq, np.abs(simUC.R_origin[index]), label="Simulated UC, V={0}".format(voltages[index]), ls='--', lw=3)
        plt.legend()
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Amplitude [1]")
        plt.title("Amplitude of the RIS reflection R")
        plt.xlim([8, 15])
        plt.ylim([0, 1.2])

    #---------------------------------------------------------------------#
    plt.show()

class TimeGatingSettings:
    def __init__(self):
        self.unit = ''
        self.mode = ''
        self.window = ''
        self.method = ''
        self.start = 0
        self.stop = 0

class MeasurementSetup:
    def __init__(self):
        self.owner       = ''
        self.date        = ''
        self.ports       = ''
        self.spacing     = ''
        self.antenna     = ''
        self.distance    = ''
        self.file        = ''
        self.param       = ''
    def getPath(self):
        self.path = 'Measurement/{0}/{1}/{2}_port_{3}_{4}_{5}mm'.format(self.owner, self.date, self.ports, self.spacing, self.antenna, self.distance)
        #    path = '../RIS_sparam/Measurement/Kyiv/2025_01_16/2_port_Hor_ant3_3300mm/'
    def getlabel(self, voltages):
        self.label = []
        for i in range(len(voltages)):
            self.label.insert(i, '{0}port {1}, inPhase RIS V={4:.1f}'.format(self.ports, self.spacing, self.antenna, self.distance, voltages[i]))
             
class readSparamSimulation():
    def __init__(self, name):
        self.path = '../RIS_BeamSteering/Simulation/UnitCell'
        self.name = name
        self.getOrigin()
        self.getComplex()

    def getOrigin(self):
        self.origin = rf.Network("{0}/{1}.s1p".format(self.path, self.name))
        return 0

    def getComplex(self):
        self.originC = self.origin.s_re + 1j*self.origin.s_im
        return 0;

class readSparamMeasurement:
    def __init__(self, setup, gate, name):
        self.setup = setup
        self.gate = gate
        self.name = name
        self.origin = 0 
        self.gated = 0
        self.originC = 0
        self.gatedC = 0

        self.getOrigin()
        self.getGated()
        self.getComplex()

    def getOrigin(self):
        network = rf.Network("{0}/{1}{2}.{3}".format(self.setup.path, self.name, self.setup.postfix, self.setup.file))
        if self.setup.param == 's11':       self.origin = network.s11
        elif self.setup.param == 's21':     self.origin = network.s21
        elif self.setup.param == 's12':     self.origin = network.s12
        elif self.setup.param == 's22':     self.origin = network.s22

    def getGated(self):
        self.gated =  self.origin.time_gate(start=self.gate.start, stop=self.gate.stop, t_unit=self.gate.unit, mode=self.gate.mode, window=self.gate.window, method=self.gate.method)
    
    def getComplex(self):
        self.originC = self.origin.s_re + 1j*self.origin.s_im
        self.gatedC = self.gated.s_re + 1j*self.gated.s_im
        return 0;

class MeasurementResult():
    def __init__(self, setup, gate, volt):
        self.setup = setup
        self.gate = gate
        self.volt = volt
        self.getDUT()
        
    def getDUT(self):
        self.noDUT = readSparamMeasurement(self.setup, self.gate, 'noDUT')
        self.metal = readSparamMeasurement(self.setup, self.gate, 'metal')
        self.DUT = []
        self.R_origin = []
        self.R_gated = []
        for i in range(len(self.volt)):
            self.DUT.insert(i, readSparamMeasurement(self.setup, self.gate, str(self.volt[i])))
            self.R_origin.insert(i, self.getR(self.DUT[i].originC, self.noDUT.originC, self.metal.originC))
            self.R_gated.insert(i,  self.getR(self.DUT[i].gatedC,  self.noDUT.gatedC,  self.metal.gatedC))
            self.R_origin[i] = self.R_origin[i].flatten()
            self.R_gated[i]  = self.R_gated[i].flatten()
        self.freq = (self.metal.origin.f)/1e9

    def getR(self, DUT, noDUT, metal):
        metal_norm = metal-noDUT
        DUT_norm = DUT-noDUT
        return -1*(DUT_norm / metal_norm)
    def setCorrection(self, correction):
        self.phase = []
        self.phase_corr = []
        N = len(correction)
        for i in range(N):
            self.phase.insert(i, np.unwrap(np.angle(self.R_gated[i]))*180/np.pi)
            self.phase_corr.insert(i, self.phase[i] + correction[i])

class SimulationResult():
    def __init__(self, volt):
        self.volt = volt
        self.getDUT()

    def getDUT(self):
        self.metal = readSparamSimulation('metal')
        self.DUT = []
        self.R_origin = []
        for i in range(len(self.volt)):
            self.DUT.insert(i, readSparamSimulation(str(self.volt[i])))
            self.R_origin.insert(i, self.getR(self.DUT[i].originC, self.metal.originC))
            self.R_origin[i] = self.R_origin[i].flatten()
        self.freq = (self.metal.origin.f)/1e9

    def getR(self, DUT, metal):
        return -1*(DUT/ metal)
    def setCorrection(self, correction):
        self.phase = []
        self.phase_corr = []
        N = len(correction)
        for i in range(N):
            self.phase.insert(i, np.unwrap(np.angle(self.R_origin[i]))*180/np.pi)
            self.phase_corr.insert(i, self.phase[i] + correction[i])

if __name__ == "__main__":
    main()

