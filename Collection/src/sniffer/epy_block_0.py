"""
BLE_Packets_Gain Block

The input data comes from GFSKDemod, which captures BLE air packets according to PREAMBLE and AccessAddress. 
According to the longest value of BLE Packets, forward the data to the next module for processing. 

Key1: Not descrambled, not calculated PDU length
Key2: Since the output length of GFSKDemod is 1024 each time, some packets may be truncated,
      and the module will choose to discard these packets.

Param1: Access Address , ADV (0x8E89BED6) Default

"""

from re import L
from struct import pack
import numpy as np
from gnuradio import gr
from array import array
import pmt


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """ BLE_Packets_Gain Block """

    def __init__(self, AA = "0x8E89BED6"):  # only default arguments here
        gr.sync_block.__init__(
            self,
            name='BLE PACKET Gain',   # will show up in GRC
            in_sig=[np.int8],
            out_sig=None
        )
        self.message_port_register_out(pmt.intern('msg_out'))

        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.AccessAddress = AA
        self.last_packets=""
        self.packets_buf = ""

    def work(self, input_items, output_items):
        bits_stream=input_items[0]
        bits_decode=""
        Octets = 257 #47

        for x in bits_stream:  
            bits_decode+=str(x)

        # Fix packets cut
        if len(self.packets_buf)!=0:
            self.packets_buf += bits_decode[:Octets-len(self.packets_buf)]
            self.message_port_pub(pmt.intern("msg_out"),pmt.intern(self.packets_buf))
            self.packets_buf = ""
            return len(input_items[0])

        if self.AccessAddress!='':
            AA =bin(int(self.AccessAddress,base=16))[2:].zfill(8*4)[::-1]
        else:
            AA=""
        PREAMBLE="01010101" # 0xaa reverse
        PREAMBLE2="10101010" # 0x55 reverse
        
        packet1 = PREAMBLE+AA
        packet2 = PREAMBLE2+AA

        if packet1 in bits_decode or packet2 in bits_decode:
            if packet1 in bits_decode:
                index = bits_decode.find(packet1)
                preamble = PREAMBLE
            else:
                index = bits_decode.find(packet2)
                preamble = PREAMBLE2
            
            if len(bits_decode) - index >= Octets*8:    # Cut pakcets
                packets = bits_decode[index:index+Octets*8]
                # 使用pmt.dict()创建消息
                msg_dict = pmt.make_dict()
                msg_dict = pmt.dict_add(msg_dict, pmt.intern("preamble"), pmt.intern(preamble))
                msg_dict = pmt.dict_add(msg_dict, pmt.intern("packets"), pmt.intern(packets))
                self.message_port_pub(pmt.intern("msg_out"), msg_dict)
            else:
                self.packets_buf = bits_decode[index:] # Fix packets
                return len(input_items[0])   

        return len(input_items[0])

    def reset_accaddr(self,aa):
        self.AccessAddress = aa