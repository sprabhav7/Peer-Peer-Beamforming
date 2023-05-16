import socket
import os
import routeros_api

# Replace these variables with your own Mikrotik RouterOS credentials and router IP address
USERNAME = "admin"
ROUTER_IP = "192.168.250.243"
PASSWORD = ""
# Get the TX sector value using the RouterOS library
def set_tx_sector(sectorno):
    # try:
    # Connect to the Mikrotik router using RouterOS API over SSH
    connection = routeros_api.RouterOsApiPool(ROUTER_IP, username=USERNAME, password=PASSWORD, plaintext_login=True)
    api = connection.get_api()
    api.get_resource('/interface/w60g').call('set',{'numbers':'0', 'tx-sector':str(sectorno)})
    # Close the RouterOS API connection
    connection.disconnect()
 
# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send a message to the server
server_address = ('192.168.250.20', 9999)
sock.bind(server_address)

while True:
    data, address=sock.recvfrom(4096)
    decodedData=data.decode()
    set_tx_sector(int(decodedData))
    #print(f'changed beam to' decodedData)
    #file1=open('data.txt','a')
    #file1.write(str(decodedData))
    #file1.close()
    print(f'Received "{data.decode()}" from {address}')
