import serial
import serial.tools.list_ports
import struct
import numpy as np
'''
CyberGlove 类用于读取手套传感器数据
并进行数据处理和映射
并将其发送到灵巧手控制器
'''
class CyberGlove:
    def __init__(self, n_df=18, s_port='/dev/ttyUSB1', baud_rate=115200, samples_per_read=1):
        self.n_df = n_df
        self.samples_per_read = samples_per_read
        self.s_port = s_port
        self.baud_rate = baud_rate
        self.__bytesPerRead = 20
        self.si = serial.Serial(port=s_port, baudrate=baud_rate, timeout=1, write_timeout=1)

    def start(self):
        if not self.si.is_open:
            self.si.open()
        self.si.reset_input_buffer()
        self.si.reset_output_buffer()

    def stop(self):
        if self.si.is_open:
            self.si.close()

    def read(self):
        fmt = '@' + "B" * self.__bytesPerRead
        data = []
        for _ in range(self.samples_per_read):
            self.si.write(b'\x47')
            msg = self.si.read(size=self.__bytesPerRead)
            if len(msg) == self.__bytesPerRead:
                raw_data = struct.unpack(fmt, msg)
                raw_data = np.asarray(raw_data)[1:-1]
                data.append(raw_data)
        #print(data)
        return np.array(data)[0] if data else None