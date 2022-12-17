# Spot Navigation in Environments with Unstructured Vegetation

This repo contains some of the scripts needed for navigating Boston Dynamics Spot in vegetation rich environments. 
These scripts depends on ROS move_base package for obtaining a local costmap.

The main script for navigation is:

**dwa_costmap.py** - Used to send velocity commands to Spot based on an updated local costmap which accounts for the pliability of certain vegetation. 