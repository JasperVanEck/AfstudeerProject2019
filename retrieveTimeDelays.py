# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:03:01 2019

@author: Jasper
"""

import sys
import numpy as np
import argparse
import qi
import time


#Code from: http://doc.aldebaran.com/2-8/naoqi/audio/alsounddetection-api.html#alsounddetection-api
def main(session):
    """
    This example uses the setParameter method.
    """
    # Get the service ALSoundDetection.

    sound_detect_service = session.service("ALSoundDetection")

    # Sets the sensitivity of the detection to 0.3 (less sensitive than default).
    # The default value is 0.9.
    sound_detect_service.setParameter("Sensitivity", 0.3)
    print("Sensitivity set to 0.3")
    
    sound_detect_service.callback()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    main(session)
