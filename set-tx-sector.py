import routeros_api
import sys
import socket
import time
import os
'''
# Replace these variables with your own Mikrotik RouterOS credentials and router IP address
USERNAME = "admin"
ROUTER_IP = "192.168.250.242"
PASSWORD = ""
'''

current_tx_sector = -1

# Main program
if __name__ == "__main__":

    pipe_path = '/home/marga3/group4/router_socket'
    pipe_fd = os.open(pipe_path, os.O_RDONLY)
    pipe_file = os.fdopen(pipe_fd)
    
    while True:
        line = pipe_file.readline()
        
        if not line:
            break
        if 'tx-sector:' in line:    
            tx_sector = line.split(':')[1]
            if (current_tx_sector == -1) or (current_tx_sector != tx_sector):
            
                current_tx_sector = tx_sector
            
                #write to file
                with open('router_file','w') as f:
                    ip_data = {
                        'sector': current_tx_sector,
                    }
                    f.write(str(ip_data))
            
        else:
            continue
            
        
            
    
    
   
