#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from munkres import Munkres, print_matrix
import sys
import os
import math
import socket
import routeros_api
import struct
import signal


# set this up
MOBILE_IP = '192.168.250.11'
FIXED_IP = '192.168.250.13'

PORT = 8080

# Create socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # # Allow script to be re-ran after quitting (close TIME_WAIT)
s.bind(('127.0.0.1', PORT))

# Listen for incoming connections
s.listen()

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print('Python program listening for connections')


# global constants wont change
eps = 0.16
min_pts = 250
alpha = 0.25
L = np.array([10000,10000,10000]) #Limit type1
L2 = 10000 #Limit type2
thresh = 5 #threshold for the dsitance above which the distance is made L2
maxFails = 10 #Maximum number of frames for which a track is checked. Failure after maxFails will lead to removal of the track

#structs monitoring tracks
global tracks
global track_status
global counter
global ip_track
global sector_val
ip_track = {MOBILE_IP:[], FIXED_IP:[]}
tracks = []
track_status = []
counter = 0
global bigdf
bigdf = pd.DataFrame()


def frame_processing(frame):
    fr =  frame
    xyz = fr[['X','Y','Z']].to_numpy()
    xyz[:,2] = 0.0 #Removing the z coordinate data
    labels = dbscan(xyz) #Performs dbscan

    doppler_array = fr[['doppler']].to_numpy()
    centroid_array = []
    centroid_doppler = []
    centroid_label = []

    labelList = set(labels)
    for x in labelList:
        if x == -1:
            continue
        points_in_cluster = xyz[labels==x,:]
        dopplers_in_cluster = doppler_array[labels==x,:]
        centroid = np.mean(points_in_cluster,axis=0)
        mean_doppler = np.mean(dopplers_in_cluster,axis=0)
        centroid_array.append(centroid)
        centroid_doppler.append(mean_doppler)
        centroid_label.append(x)
    # if not (centroid_array or centroid_label):
    #     continue
    ####*** Need to see what happens if no centroids exist in a frame ***#####
    tracks_update(centroid_array,centroid_doppler,centroid_label)
    
def dbscan(xyz):
    scaled_points = StandardScaler().fit_transform(xyz)
    #print(f"Scaled points are : Max = {max(scaled_points[0])}, Min = {min(scaled_points[0])}")
    ##Clustering frame using dbscan
    model = DBSCAN(eps= eps,min_samples= min_pts)
    model.fit(scaled_points)
    return model.labels_

'''
Centroid doppler and label not being used
'''
def tracks_update(centroid_array, centroid_doppler,centroid_label):

    global tracks
    global track_status
    global counter

    #print("\nFrame number = ", counter)
    counter += 1
    if not tracks:
        for coordinate in centroid_array:
            tracks.append([coordinate])
        track_status = [0]*len(tracks)
        return
    
    #Pick the last coordinates from each track. Also, convert current frame to a list of numpy arrays
    track_points = extract_last_points(tracks) #1. Last coordinates from track
    fr_points = convert(centroid_array) #2. Convert the numpy array of array to lsit of numpy arrays

    #print(f"Frame {counter} statistics:")
    #print(f"tracks length = {len(tracks)}")
    #print(f"fr_points length = {len(fr_points)}")
    #print(f"This set points must be returned = {fr_points}?")

    #Padding the shorter array with L, where L is a far away coordinate wrt all axes.
    lendiff = len(track_points) - len(fr_points)
    appender = [L]*abs(lendiff)
    if lendiff > 0:
        fr_points.extend(appender)
    elif lendiff < 0:
        track_points.extend(appender)

    #Create matrix of distances from the two lists
    matrix = dist_matrix(track_points,fr_points)
    
    #Applying Hungarian algorithm
    m = Munkres()
    indexes = m.compute(matrix)

    for row, column in indexes:
        d = matrix[row][column]
        #print(f'({row}, {column}) -> {d}')
        
        #update tracks happens here
        if d < L2:
            #Adding the point fr_points[column] to the tracks[row]
            tracks[row].append(fr_points[column])
            
            # update the track for corresponding ip
            update_ip_tracks(row, fr_points[column])
    
    # delete tracks happens here
    for row, column in indexes:
        d = matrix[row][column]
        if d >= L2:
            #1. Process the tracks and track_status arrays.
            update_fail_tracks(row)
            #2. Add the cluster from the current frame as a new track.
            add_to_tracks(fr_points[column],column)
    #Delete tracks that are marked for deletion
    if len(tracks) > 0:
        delete_tracks()
    #Visualizing
    #visualize()
    

def dist(p1, p2):
    #Calculating distance between two points. Z-axis is weighted to reduce importance.
    if all(p1 == L) or all(p2 == L):
        return L2
    distance = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 +  alpha*(p1[2]-p2[2])**2
    return distance

def extract_last_points(arr):
    return [item[-1] for item in arr]

def convert(arr):   
    lister = [] 
    for x in arr:
        lister.append(x)
    return lister

def dist_matrix(m,n):
    #Create distance matrix from two arrays
    mat = []
    for i in range(len(m)):
        row = []
        for j in range(len(n)):
            row.append(dist(m[i],n[j]))
        mat.append(row)
    mat = np.array(mat)
    #Thresholding the matrix so that any value above thresh is made L2
    mat[mat > thresh] = L2
    mat = mat.tolist()
    return mat

def update_fail_tracks(ind):
    if ind < len(track_status):
        track_status[ind] += 1

def delete_tracks():
    i = 0
    length = len(track_status)
    #If failure happens for maxFails times
    while i < length:
        if track_status[i] >= maxFails:
            del tracks[i]
            del track_status[i]
            i -= 1
            length -= 1
        i += 1

def add_to_tracks(point,column):
    #If the point was an augmented point, ignore
    if all(point == L):
        return
    #Else add point as a new track
    tracks.append([point])
    track_status.append(0)

def visualize( ):
    printset_master = []
    print(f"Tested {counter} frames")
    labelind = 0
    printset = np.array(tracks[0][-1])
    # print("printset is ",printset)
    printlabels = np.array([labelind])#*len(tracks[0]))
    # print("printlabels is ",printlabels)
    i = 0
    for row in tracks:
        if i == 0:
            i +=1
            continue
        printset = np.vstack([printset,np.array(row[-1])])
        labelind+=1
        printlabels = np.append(printlabels, [labelind])
    printset = np.vstack([printset,np.array([0,0,0])])
    print("printset is \n", printset)
    pcd_track = o3d.geometry.PointCloud()
    pcd_track.points = o3d.utility.Vector3dVector(printset)
    numClusters = len(printlabels)
    colors_track = plt.get_cmap("tab10")(printlabels/ (numClusters if numClusters > 0 else 1))
    pcd_track.colors= o3d.utility.Vector3dVector(colors_track[:,:3])

    printset_master.append(printset)
    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0, 0, 0])
    # o3d.visualization.RenderOption(point_size=2)
    print(f"Length of tracks is {len(tracks)}")
    o3d.visualization.draw_geometries([pcd_track], mesh_show_wireframe=True)

def update_ip_tracks(row,column_data):
    global tracks
    global ip_track
    for key in ip_track.keys():
        if np.array_equal(ip_track[key],tracks[row]):
            tracks[row].append(column_data)
            ip_track[key] = tracks[row]
            
            #Dont update anything below this
            #Sending data to mobile ip about others coordinates
            if len(ip_track[MOBILE_IP])>0 and len(ip_track[FIXED_IP])>0:
                send_data_to_mobAP()

def send_data_to_mobAP():
    # Send a message to the server
    server_address = (MOBILE_IP, 9999)
    # message = str({'192.168.250.11' : ip_track['192.168.250.11'][-1],'192.168.250.13' : ip_track['192.168.250.13'][-1]})
    # message = str(ip_track['192.168.250.11'][-1][0])+","+str(ip_track['192.168.250.11'][-1][1])+","+str(ip_track['192.168.250.13'][-1][0])+","+str(ip_track['192.168.250.13'][-1][1])
    message = str(ip_track[MOBILE_IP][-1][0])+","+str(ip_track[MOBILE_IP][-1][1])+",-2.01,2.32"
    sock.sendto(message.encode(), server_address)
    # Wait for a response from the server
    # response, server = sock.recvfrom(4096)

    # Print the response from the server
    # print(f'Received "{response.decode()}" from {server}')

def getTxSector(alpha):
    if alpha > 22.8:
        txSector = 24
    elif alpha <= 22.8 and alpha > 15.2:
        txSector = 25
    elif alpha <= 15.2 and alpha > 7.6:
        txSector = 26
    elif alpha <= 7.6 and alpha > 0:
        txSector = 27
    elif alpha < -22.8:
        txSector = 31
    elif alpha >= -22.8 and alpha < -15.2:
        txSector = 30
    elif alpha >= -15.2 and alpha < -7.6:
        txSector = 29
    elif alpha >= -7.6 and alpha <= 0:
        txSector = 28
    	
    return txSector

def is_connected(hostname):
    try:
        # see if we can resolve the host name -- tells us if there is
        # a DNS listening
        host = socket.gethostbyname(hostname)
        # connect to the host -- tells us if the host is actually reachable
        s = socket.create_connection((host, 9999), 2)
        s.close()
        return True
    except Exception:
        pass # we ignore any errors, returning False
    return False

def identify_cluster_ip(sector_p_track,tan_inv_degrees):
    for angle in tan_inv_degrees:
        if angle > 0 and len(ip_track[MOBILE_IP]) <= 0:
            #Assign it the ip address 192.168.250.11 == MOBILE_IP
            ip_track[MOBILE_IP] = tracks[tan_inv_degrees.index(angle)]
        elif angle < 0 and len(ip_track[FIXED_IP]) <= 0:
            #Assign it the ip address 192.,168.250.13 == FIXED_IP
            ip_track[FIXED_IP] = tracks[tan_inv_degrees.index(angle)]
        else:
            # future implementation
            pass
    
    print('Assigned')


def get_codebook_sector():
    sector_per_track = list()
    #print(f'Tracks : {tracks}')
    tan_inv_degrees = [90-math.degrees(np.arctan2(t[-1][1],t[-1][0])) for t in tracks]
    print(f'Tan inv degrees : {tan_inv_degrees}')
    #print(f'From tan-1 deg : {tan_inv_degrees}')
    # I am trying to find the minimum indices for all the angles in tan_inv_degrees
    for tan_inv_degree in tan_inv_degrees:
        sector_val = getTxSector(tan_inv_degree)
        sector_per_track.append(sector_val)    

    # from that i get the codebook sectors
    print(f'Corresponding Tx Sector : {sector_per_track}')

    # set tx sector, then send message asking for ip
    # set the ip track dict if i get resp

    if len(ip_track[MOBILE_IP]) <= 0 or len(ip_track[FIXED_IP]) <= 0:
        identify_cluster_ip(sector_per_track, tan_inv_degrees)
    ip_track_print = {}
    for k,v in ip_track.items():
        if not v:
            ip_track_print[k] = v
        else:
            ip_track_print[k] = v[-1]
    print(f'IP Track = {ip_track_print}')
        
def signal_handler(sig,frame):
    print('Pressed')
    bigdf.to_csv('mycluster.csv',index=False)
    exit(0)


if __name__ == '__main__':
    # load up the sector_val at the start
    # if frame start processing, else keep reading
    # pipe_path = '/home/marga3/group4/Beams-main/Apr24/test.csv'
    # pipe_fd = os.open(pipe_path, os.O_RDONLY)
    # pipe_file = os.fdopen(pipe_fd)
    # csv_path = 'clusterdata'
    # pipe_file = open(csv_path)
    try:
        print('Pipe opened, lets hope nothing blows up')
        while True:
            # Accept incoming connection
            conn, addr = s.accept()
            #print('Connection established from', addr)
        
            # Receive data from connection
            #data = conn.recv(100016)
            #print('Data received in Python:', data)


            # Unpack the binary data into a list of structs

            vector_size= struct.unpack('Q', conn.recv(struct.calcsize('Q')))[0]

            vector_data =b''

            while len(vector_data)< vector_size * struct.calcsize('fffffii'):
                vector_data += conn.recv(vector_size * struct.calcsize('fffffii')- len(vector_data))


            received_vector=[]
            for i in range(vector_size):

                point = struct.unpack_from('fffffii', vector_data, i * struct.calcsize('fffffii'))

                received_vector.append(point)
            
            df = pd.DataFrame(received_vector, columns=['X','Y','Z','power','doppler','frame','point'])
            #bigdf = pd.concat([bigdf,df])
            frame_processing(df)
            if len(tracks) > 0:
                    get_codebook_sector()
    except KeyboardInterrupt:
        print('Pressed')
        #bigdf.to_csv('mycluster.csv',index=False)
        exit(0)
        
    # Close connection
    conn.close()
