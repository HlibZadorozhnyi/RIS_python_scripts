import scipy
from scipy.interpolate import RBFInterpolator, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import skrf as rf
import numpy as np

def main():

    rf.stylely()

    voltages = [0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19.8]

    setup = MeasurementSetup()
    setup.owner       = 'Kyiv'            # owner of the S-parameters: Kyiv or Braunschweig
    setup.date        = '2025_01_16'      # Date of the measurement
    setup.ports       = '1'               # 1 or 2 port setup
    setup.spacing     = 'None'            # spacing of the 2 port setup ['Hor', 'Ver']
    setup.antenna     = 'ant3'            # antenna for the measurement ['ant1', 'ant2', 'ant3', 'openWR90']
    setup.distance    = '3300'            # distance of the measurement in [mm]
    setup.file        = 's1p'             # depends on the measurement system: s1p or s2p
    setup.param       = 's11'             # select the Sparam to apply the Time Analyzis
    setup.postfix     = ''                # optional for Braunschweig ['_1', '_2', ..., '_7', '_8']
    setup.getPath()
    setup.getlabel(voltages)

    gate = TimeGatingSettings()
    gate.start = 22
    gate.stop = 25
    gate.unit = 'ns'
    gate.mode = 'bandpass'
    gate.window = ('kaiser',5) # 'boxcar', 'hann' or 'hamming'
    gate.method = 'fft'

    meas = MeasurementResult(setup, gate, voltages)

    #---------------------------------------------------------------------#
    # Estimate the UnitCell`s Phase Span, BandWidth, Fmin, F0, Fmax
    uc = UnitCell(meas)

    plt.figure(1)       
    uc.plotPhaseSpan()

    plt.figure(2)
    uc.plotPhaseAmplitude2Voltage() 

    # #---------------------------------------------------------------------#
    # # Estimate the RIS Phase Profile and Voltages across the Unit Cells
    bf = BeamForming()
    AF = AntennaFactor(bf)
    RIS = RIS_plate(uc, bf, AF)

    plt.figure(3)
    RIS.plotUnitCell_PhaseVoltage()
    
    #---------------------------------------------------------------------#
    # Estimate Antenna Factor
    AF.get_Balanis_3D_AntennaFactor(RIS)

    plt.figure(4)
    AF.plotAntennaFactor()
    
    # #---------------------------------------------------------------------#
    # # Show the Results
    print("Working frequency range from Fmin={0:.2f}GHz to Fmax={1:.2f}GHz".format(uc.Fmin, uc.Fmax))
    print("Central frequency F0={0:.2f}GHz".format(uc.F0))
    print("Bandwidth BW={0:.2f}GHz or BW={1:.2f}%".format(uc.BW, (uc.BW/uc.F0)*100))

    print("")
    print("Voltages to set in the RIS:")
    for i in range(RIS.Nx):
        print("Column [{0}]: \t {1:.2f}V ".format(i+1, RIS.voltage[i]))


    plt.show()

def rad2deg(rad):
    return rad*180/np.pi

def deg2rad(deg):
    return deg*np.pi/180


class AntennaFactor:
    def __init__(self, bf):
        self.bf = bf
        self.PhiR_limit_deg = []
        self.PhiR_ideal_deg = []
        self.PhiR_limit_rad = []
        self.PhiR_ideal_rad = []
    
    def get_Balanis_3D_AntennaFactor(self, RIS):
        self.theta_deg = np.linspace(-90, 90, 1801)
        # self.phy_deg   = np.linspace(-180, 180, ((180*2)+1))
        self.phy_deg   = np.linspace(0, 0, 1)
        self.theta_rad = deg2rad(self.theta_deg)
        self.phy_rad   = deg2rad(self.phy_deg)

        self.N_theta = len(self.theta_deg)
        self.N_phy   = len(self.phy_deg)

        self.ideal = np.zeros((self.N_theta, self.N_phy), dtype=complex)
        self.limit = np.zeros((self.N_theta, self.N_phy), dtype=complex)

        max_theta = self.theta_deg[np.argmax(self.theta_deg)]

        for i in range(self.N_theta):
            print("Theta={0:.2f}\tTheta_max={1:.2f}".format(self.theta_deg[i], max_theta), end='\r')
            for j in range(self.N_phy):

                for y in range(RIS.Ny):
                    for x in range(RIS.Nx):
                        # A = np.exp(-1*1j*self.PhiI_rad[y][x]);
                        # B = np.exp(-1*1j*RIS.phase_ideal_deg[y][x]);

                        C = 1*RIS.k0*RIS.dX*(x*np.sin(self.theta_rad[i])*np.cos(self.phy_rad[j]) + y*np.sin(self.theta_rad[i])*np.sin(self.phy_rad[j]));

                        self.ideal[i][j] = self.ideal[i][j] +           1*np.exp(-1*1j*(self.PhiR_ideal_rad[y][x]+C))
                        self.limit[i][j] = self.limit[i][j] + RIS.ampl[x]*np.exp(-1*1j*(self.PhiR_limit_rad[y][x]+C))

                self.ideal[i][j] = self.ideal[i][j]/(RIS.Nx*RIS.Ny)
                self.limit[i][j] = self.limit[i][j]/(RIS.Nx*RIS.Ny)

        self.limit_dB = 20*np.log10(abs(self.limit))
        self.ideal_dB = 20*np.log10(abs(self.ideal))

    def plotAntennaFactor(self):
        plt.plot(self.theta_deg, self.limit_dB, label="AF limited")
        plt.plot(self.theta_deg, self.ideal_dB, label="AF ideal")
        plt.axvline(self.bf.thetaR_deg, color="black", linestyle="--", label="Reflected Beam={0}deg".format(self.bf.thetaR_deg))
        plt.legend()
        plt.xlabel("Azimuth [deg]")
        plt.ylabel("Magnitude [dB]")
        plt.ylim([-60, 0])

class RIS_plate:
    def __init__(self, uc, bf, AF):
        self.uc = uc
        self.bf = bf
        self.AF = AF
        self.F0 = self.uc.Fris
        self.dX = 10
        self.Nx = 18
        self.dY = 10
        self.Ny = 23
        self.L0 = (scipy.constants.c*1e-6)/self.F0
        self.k0 = (2*np.pi)/self.L0
        self.MaxPhase_deg = uc.phase_i[uc.phase_i.argmax()]

        self.get_dPhy()
        self.getPhaseProfile()
        self.getVoltageAtUnitCell()

    def get_dPhy(self):
        if(self.bf.thetaI_deg == 0):
            dL = self.dX*np.sin(self.bf.thetaR_rad); 
            self.dPhy_deg = -1*360*dL/self.L0;
            self.dPhy_rad = deg2rad(self.dPhy_deg)
        else:
            R = self.bf.nR*np.sin(self.bf.thetaI_rad);
            I = self.bf.nI*np.sin(self.bf.thetaR_rad);
            self.dPhy_rad = self.dX*self.k0*(R-I);
            self.dPhy_deg = rad2deg(self.dPhy_rad)

    def getPhaseProfile(self):

        self.phase_limit_deg = np.zeros((self.Ny, self.Nx))
        self.phase_ideal_deg = np.zeros((self.Ny, self.Nx))

        self.PhiR_rad = np.zeros((self.Ny, self.Nx))
        self.PhiI_rad = np.zeros((self.Ny, self.Nx))

        self.PhiR_deg = np.zeros((self.Ny, self.Nx))
        self.PhiI_deg = np.zeros((self.Ny, self.Nx))

        self.AF.PhiR_limit_deg = np.zeros((self.Ny, self.Nx))
        self.AF.PhiR_ideal_deg = np.zeros((self.Ny, self.Nx))

        self.midVal = (360-self.MaxPhase_deg)/2;

        for y in range(self.Ny):
            for x in range(self.Nx):

                self.PhiR_rad[y][x] = -1*self.k0*self.dX*(x*np.sin(self.bf.thetaR_rad)*np.cos(self.bf.phyR_rad) + y*np.sin(self.bf.thetaR_rad)*np.sin(self.bf.phyR_rad))
                self.PhiI_rad[y][x] = +1*self.k0*self.dX*(x*np.sin(self.bf.thetaI_rad)*np.cos(self.bf.phyI_rad) + y*np.sin(self.bf.thetaI_rad)*np.sin(self.bf.phyI_rad))

                self.PhiR_deg[y][x] = rad2deg(self.PhiR_rad[y][x])
                self.PhiI_deg[y][x] = rad2deg(self.PhiI_rad[y][x])

                # Y = (self.PhiR_deg[y][x] - self.PhiI_deg[y][x]) - self.dPhy_deg;
                Y = (self.PhiR_deg[y][x] - self.PhiI_deg[y][x])

                while Y>=360:
                    Y = Y-360

                if (Y>self.MaxPhase_deg) and (Y<360) and ((Y-self.MaxPhase_deg)<self.midVal) :
                    self.phase_limit_deg[y][x] = self.MaxPhase_deg
                    self.phase_ideal_deg[y][x] = Y
                elif (Y>self.MaxPhase_deg) and (Y<360) and ((360-Y)<=self.midVal) :
                    self.phase_limit_deg[y][x] = 0
                    self.phase_ideal_deg[y][x] = Y
                else:
                    self.phase_limit_deg[y][x] = Y
                    self.phase_ideal_deg[y][x] = Y

                self.AF.PhiR_ideal_deg[y][x] = self.phase_ideal_deg[y][x] + self.PhiI_deg[y][x]
                self.AF.PhiR_limit_deg[y][x] = self.phase_limit_deg[y][x] + self.PhiI_deg[y][x]

        self.phase_limit_rad = deg2rad(self.phase_limit_deg)
        self.phase_ideal_rad = deg2rad(self.phase_ideal_deg)
        self.AF.PhiR_limit_rad = deg2rad(self.AF.PhiR_limit_deg)
        self.AF.PhiR_ideal_rad = deg2rad(self.AF.PhiR_ideal_deg)

    def getVoltageAtUnitCell(self):
        self.voltage = np.empty(self.Nx)
        self.voltage_index = np.empty(self.Nx)
        self.ampl = np.empty(self.Nx)
        for i in range(self.Nx):
            diff = abs(self.uc.phase_i - self.phase_limit_deg[1][i]);
            index = np.argmin(diff);
            self.voltage_index[i] = index;
            self.voltage[i] = self.uc.volt_i[index];
            self.ampl[i] = self.uc.amplitude_i[index];

    def plotUnitCell_PhaseVoltage(self):
        plt.subplot(1,2,1)
        plt.stem(np.linspace(1, self.Nx, num=self.Nx), self.voltage)
        plt.title("Voltages across the the unit cells")
        plt.xlabel("Unit Cell [1]")
        plt.ylabel("Voltage [V]")

        plt.subplot(1,2,2)
        plt.plot(np.linspace(1, self.Nx, num=self.Nx), self.phase_ideal_deg[1][:], color="green", label="Ideal phase")
        plt.stem(np.linspace(1, self.Nx, num=self.Nx), self.phase_limit_deg[1][:], label="Limited phase")
        plt.axhline(self.MaxPhase_deg, color="black", linestyle="--", label="Maximum Phase={0:.2f}deg".format(self.MaxPhase_deg))
        plt.legend()
        plt.title("Reflected Phase across the unit cells")
        plt.xlabel("Unit Cell [1]")
        plt.ylabel("Phase [deg]")

class BeamForming:
    def __init__(self):
        self.thetaR_deg = -20
        self.thetaI_deg = 0
        self.phyR_deg   = 0
        self.phyI_deg   = 0

        self.thetaR_rad = deg2rad(self.thetaR_deg)
        self.thetaI_rad = deg2rad(self.thetaI_deg)
        self.phyR_rad   = deg2rad(self.phyR_deg)
        self.phyI_rad   = deg2rad(self.phyI_deg)

        self.nI = 1
        self.nR = 1

class UnitCell:
    def __init__(self, meas):
        self.meas = meas
        self.volt_i = np.linspace(0.01, 19.8, int((19800-10)/10))
        self.N_volt_i = len(self.volt_i)
        self.volt = self.meas.volt
        self.N_volt = len(self.meas.volt)
        self.phaseSpan_Desired = 240
        self.Fris = 0

        self.getPhaseSpan()
        self.getFminF0FmaxBW()
        self.Fris = self.F0
        self.getPhase2Voltage(self.Fris)
        self.getAplitude2Voltage(self.Fris)
        self.getCostFunctionParams()   

    def getAplitude2Voltage(self, f):
        diff = np.abs(self.meas.freq - f)
        index = np.argmin(diff)
        self.amplitude = np.abs(np.take(self.meas.R_origin, indices=index, axis=1))

    def getPhase2Voltage(self, f):
        diff = np.abs(self.meas.freq - f)
        index = np.argmin(diff)
        phase = np.angle(np.take(self.meas.R_gated, indices=index, axis=1))
        phase = np.unwrap(phase)*180/np.pi
        self.phase = phase - phase[0]

    def getPhaseSpan(self):
        phaseMax = (np.unwrap(np.angle(self.meas.R_gated[self.N_volt-1]))*180/np.pi)+360
        phaseMin = np.unwrap(np.angle(self.meas.R_gated[0]))*180/np.pi
        self.phaseSpan = phaseMax - phaseMin

    def getFminF0FmaxBW(self):
        fstart = 8
        fstop  = 14

        index_fstart = int(np.where(self.meas.freq == fstart)[0])
        index_fstop = int(np.where(self.meas.freq == fstop)[0])

        phaseSpan = self.phaseSpan[index_fstart:index_fstop]
        freq = self.meas.freq[index_fstart:index_fstop]

        diff = np.abs(phaseSpan - self.phaseSpan_Desired)
        index3 = np.argmax(phaseSpan)
        index1 = np.argmin(diff)
        if index1<index3:
            self.Fmin = freq[index1]
            diff[index1] = 360
            index2 = np.argmin(diff)
            self.Fmax = freq[index2]
        elif index1>index3:
            self.Fmax = freq[index1]
            diff[index1] = 360
            index2 = np.argmin(diff)
            self.Fmin = freq[index2]
        
        self.BW = self.Fmax - self.Fmin
        self.F0 = freq[index3]

    def getCostFunctionParams(self):
        ius = InterpolatedUnivariateSpline(self.volt, self.phase)
        self.phase_i = ius(self.volt_i)
        ius = InterpolatedUnivariateSpline(self.volt, self.amplitude)
        self.amplitude_i = ius(self.volt_i)

        self.Amin = self.amplitude_i[np.argmin(self.amplitude_i)]
        self.Amax = self.amplitude_i[np.argmax(self.amplitude_i)]
        self.Aave = np.average(self.amplitude_i)
        self.Adel = self.Amax - self.Amin

    def plotPhaseSpan(self):
        plt.plot(self.meas.freq, self.phaseSpan, label="Obtained")
        plt.axhline(y=self.phaseSpan_Desired, color="black", linestyle="--", label="Desired={0}".format(self.phaseSpan_Desired))
        plt.axvline(x=self.Fmin, color="red",   linestyle=":", label="Fmin={0:.2f}GHz".format(self.Fmin))
        plt.axvline(x=self.F0,   color="green", linestyle=":", label="F0={0:.2f}GHz".format(self.F0))
        plt.axvline(x=self.Fmax, color="blue",  linestyle=":", label="Fmax={0:.2f}GHz".format(self.Fmax))
        plt.legend()
        plt.xlim([9,14])
        plt.title("Phase Span")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("Phase[deg]")
        

    def plotPhaseAmplitude2Voltage(self):
        plt.subplot(1,2,1)
        plt.plot(self.volt, self.phase, label="Obtained")
        plt.plot(self.volt_i, self.phase_i, label="Interpolated")
        plt.legend()
        plt.ylim([0,360])
        plt.title("Phase at F={0:.2f}GHz".format(self.Fris))
        plt.xlabel("Bias Voltage [V]")
        plt.ylabel("Phase [deg]")

        plt.subplot(1,2,2)
        plt.plot(self.volt,   self.amplitude,   label="Obtained")
        plt.plot(self.volt_i, self.amplitude_i, label="Interpolated")
        plt.axhline(y=self.Amin, color="red",    linestyle="--", label="Amin={0:.2f}".format(self.Amin))
        plt.axhline(y=self.Amax, color="green",  linestyle="--", label="Amax={0:.2f}".format(self.Amax))
        plt.axhline(y=self.Aave, color="blue",   linestyle="--", label="Aave={0:.2f}".format(self.Aave))
        plt.legend()
        plt.ylim([0,1])
        plt.title("Amplitude at F={0:.2f}GHz".format(self.Fris))
        plt.xlabel("Bias Voltage [V]")
        plt.ylabel("Amplitude [1]")

class SettingsMeasurement:
    def __init__(self, Sowner, Sparam, Sfile, PathMeas):
        self.sowner = Sowner
        self.sparam = Sparam
        self.sfile = Sfile
        self.pathMeas = PathMeas
        self.path = ''
        self.postfix = ''

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

if __name__ == "__main__":
    main()

