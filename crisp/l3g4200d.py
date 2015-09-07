# -*- coding: utf-8 -*-
from __future__ import division

"""
Created on Wed Mar  5 10:24:38 2014

@author: hannes
"""

import numpy as np
import struct

class ParserException(Exception):
    pass

class GyroParserBase(object):
    def __init__(self):
        self.fs = None # Full scale resolution. Max rate. (dps)
        self.data_scale = None # mdps per digit
        self.data = None

    def parse(self, data):
        raise NotImplemented()
        
class L3GArduinoParser(GyroParserBase):
    COMMAND_START = 0xAB
    COMMAND_DATA  = 0xC8
    COMMAND_SAMPLE_RATE = 0xDB
    COMMAND_TIME_SYNC = 0xBB
    REG_CTRL1 = 0x20
    REG_CTRL4 = 0x23
    FS_RATE_FACTOR = { 250 : 8.75e-3, # From L3G datasheet
                       500 : 17.50e-3, 
                       2000 : 70e-3 }
    
    def __init__(self):
        GyroParserBase.__init__(self)
        self.reg = {} # Register -> value map
        self.ndata = 0
        self.actual_data_rate = None
        self.fs = None
        self.data_scale = None
        self.sync_times = []
    
    def parse(self, input_data):
        total_bytes = len(input_data)
        self.data = np.empty((3, total_bytes // 6)) # Will be lower than this
        temp_data = input_data
        num_bytes = 0
        while num_bytes < total_bytes - 1:
             command = ord(temp_data[num_bytes])
             length = ord(temp_data[num_bytes + 1])
             data = temp_data[num_bytes+2:num_bytes+2+length]
             #data_str = " ".join(["%x" % ord(x) for x in data])
             #print "Command: %x, Length: %x, Data: %s" % (command, length, data_str)
             self.__handle(command, data)
             num_bytes += length + 2
        try:
            self.data *= self.data_scale
        except RuntimeWarning:
            print "Scale warning! res=", self.data, "*", self.data_scale
        self.data = self.data[:,0:self.ndata]
             
                    
    def __handle(self, command, data):
        if command == L3GArduinoParser.COMMAND_DATA:
            if not self.data_scale:
                raise ParserException("No data scale loaded before first data packet")
            raw_str = data #b''.join([chr(x) for x in data])
            sfmt = "<hhh"
            n = len(raw_str) // 6
            for i in range(n):
                data_str = raw_str[6*i:6*i+6]
                x, y, z = struct.unpack(sfmt, data_str)
                arr= np.array([x, y, z])#.reshape(3, 1)
                self.data[:, self.ndata] = arr
                self.ndata += 1
        elif command == L3GArduinoParser.COMMAND_START:
            for reg, val in zip(data[::2], data[1::2]):
                reg = ord(reg)
                val = ord(val)
                self.reg[reg] = val
                #print "Checking reg %x with val %x" % (reg, val)
                if reg == L3GArduinoParser.REG_CTRL4:
                    fsbits = (val & 0x30) >> 4
                    fs = {0 : 250, 1 : 500, 2 : 2000, 3 : 2000}[fsbits]
                    self.fs = fs
                    self.data_scale = L3GArduinoParser.FS_RATE_FACTOR[self.fs]
                if reg == L3GArduinoParser.REG_CTRL1:
                    drbits = (val & 0xC0) >> 6
                    self.data_rate = {0 : 100, 1 : 200, 2 : 400 , 3 : 800}[drbits]
        elif command == L3GArduinoParser.COMMAND_SAMPLE_RATE:
            sfmt = '<L'
            #print "Sample rate byte (%d) %s" % (len(data), data.__repr__())
            #T = struct.unpack(sfmt, data)
            #print "Got sample rate", T
            print "Got sample rate byte, but until implementation is changed in Arduino code, the value should not be used as it is unstable as hell."
            # Note: Reimplement the Arduino code to emit timestamps on a regular basis that can be used to fix the time rate.
            self.actual_data_rate = None#1000000. / T[0] # Hz
        elif command == L3GArduinoParser.COMMAND_TIME_SYNC:
            sfmt = '<L'
            ts = struct.unpack(sfmt, data)[0] / 1000.0 # msec -> seconds
            self.sync_times.append((self.ndata, ts))
                    
def load_L3G_arduino(filename, remove_begin_spurious=False, return_parser=False):
    "Load gyro data collected by the arduino version of the L3G logging platform, and return the data (in rad/s), a time vector, and the sample rate (seconds)"
    file_data = open(filename, 'rb').read()
    parser = L3GArduinoParser()
    parser.parse(file_data[7:]) # Skip first "GYROLOG" header in file
    data = parser.data
    if parser.actual_data_rate:
        T = 1. / parser.actual_data_rate
        print "Found measured data rate %.3f ms (%.3f Hz)" % (1000*T, 1. / T)
    else:
        T = 1. / parser.data_rate
        print "Using data rate provided by gyro (probably off by a few percent!) %.3f ms (%.3f Hz)" % (1000*T, 1. / T)
        
    N = parser.data.shape[1]
    t = np.linspace(0, T*N, num=data.shape[1])
    print t.shape, data.shape
    print "Loaded %d samples (%.2f seconds) with expected sample rate %.3f ms (%.3f Hz)" % (N, t[-1], T*1000.0, 1./T)
    try:
        print "Actual sample rate is %.3f ms (%.3f Hz)" % (1000. / parser.actual_data_rate, parser.actual_data_rate, )
    except TypeError:
        pass
    
    if remove_begin_spurious:
        to_remove = int(0.3/T) # Remove first three tenth of second
        data[:,:to_remove] = 0.0
    
    if return_parser:
        return np.deg2rad(data), t, T, parser
    else:
        return np.deg2rad(data), t, T

def post_process_L3G4200D_data(data, do_plot=False):
    def notch(Wn, bandwidth):
        f = Wn/2.0
        R = 1.0 - 3.0*(bandwidth/2.0)
        K = ((1.0 - 2.0*R*np.cos(2*np.pi*f) + R**2)/(2.0 -
        2.0*np.cos(2*np.pi*f)))
        b,a = np.zeros(3),np.zeros(3)
        a[0] = 1.0
        a[1] = - 2.0*R*np.cos(2*np.pi*f)
        a[2] = R**2
        b[0] = K
        b[1] = -2*K*np.cos(2*np.pi*f)
        b[2] = K
        return b,a

    # Remove strange high frequency noise and bias
    b,a = notch(0.8, 0.03)
    data_filtered = np.empty_like(data)
    from scipy.signal import filtfilt
    for i in range(3):
        data_filtered[i] = filtfilt(b, a, data[i])

    if do_plot:
        from matplotlib.pyplot import subplot, plot, specgram, title
        # Plot the difference
        ax = None
        for i in range(3):
            if ax is None:
                ax = subplot(5,1,i+1)
            else:
                subplot(5,1,i+1, sharex=ax, sharey=ax)
            plot(data[i])
            plot(data_filtered[i])
            title(['x','y','z'][i])
        subplot(5,1,4)
        specgram(data[0])
        title("Specgram of biased X")
        subplot(5,1,5)
        specgram(data_filtered[0])
        title("Specgram of filtered unbiased X")

    return data_filtered
