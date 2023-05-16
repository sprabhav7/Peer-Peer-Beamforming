# Peer-Peer-Beamforming


Goal: With the assistance of RADAR and IMU data, the project aims to allow mmWave routers to beamform with each other without the overhead of beamscanning

Description: 
mmWaves routers while providing high bandwidth are limited in operational capability by the directionality and attenuation of mmWaves. Therefore, these routers rely on beam forming with clients to ensure high SNR and throughput. Beamforming is normally initiated in two steps
1. Coarse grain beam scanning by the router through a quasi-omni beam
2. Sector sweeping to refine the beam to a fine grained highly directional beam 

Therefore, by clustering RADAR data and estimating location/AoA from the IMU, we are able to directly provide the mmWave clients with the angle and tx-sector values to initiate beam forming thereby obviating beam scanning.
