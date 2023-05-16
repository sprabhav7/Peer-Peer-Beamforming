#!/bin/bash
source ./devel/setup.bash
/opt/ros/noetic/bin/roslaunch ./src/retina4sn_viewer/share/srs_4sn.launch output_file:=/home/marga3/work/data/t.csv
# /opt/ros/noetic/bin/rosclean purge -y
