# -*- coding: utf-8 -*-

# from: https://community.aldebaran.com/static/uploads/Alexandre/retrieve_robot_audio_buffer.py


###########################################################
# Retrieve robot audio buffer
# Syntaxe:
#    python scriptname --pip <ip> --pport <port>
#
#    --pip <ip>: specify the ip of your robot (without specification it will use the NAO_IP defined some line below
#
# Author: Alexandre Mazel
###########################################################

NAO_IP = "10.0.252.126" # Romeo on table
NAO_IP = "10.0.253.99" # Nao Alex Blue


from optparse import OptionParser
import naoqi
import numpy as np
import time
import sys
import scipy
import motion
from scipy.fftpack import rfft, dct
from sklearn.externals import joblib


class SoundReceiverModule(naoqi.ALModule):
    """
    Use this object to get call back from the ALMemory of the naoqi world.
    Your callback needs to be a method with two parameter (variable name, value).
    """

    def __init__( self, strModuleName, strNaoIp ):
        try:
            naoqi.ALModule.__init__(self, strModuleName )
            self.BIND_PYTHON( self.getName(),"callback" )
            self.strNaoIp = strNaoIp
            self.outfile = None
        except Exception, e:
            print( "ERR: abcdk.naoqitools.SoundReceiverModule: loading error: %s" % str(err) )

    # __init__ - end
    def __del__( self ):
        print( "INF: abcdk.SoundReceiverModule.__del__: cleaning everything" )
        self.stop()


    def start( self ):
        global counter, lastCounter, audio
        self.counter = 0
        self.lastCounter = 0

        audio = naoqi.ALProxy( "ALAudioDevice", self.strNaoIp, 9559 );
        nNbrChannelFlag = 0; # ALL_Channels: 0,  AL::LEFTCHANNEL: 1, AL::RIGHTCHANNEL: 2; AL::FRONTCHANNEL: 3  or AL::REARCHANNEL: 4.
        nDeinterleave = 0;
        nSampleRate = 48000;
        audio.setClientPreferences( self.getName(),  nSampleRate, nNbrChannelFlag, nDeinterleave ); # setting same as default generate a bug !?!
        audio.enableEnergyComputation()
        audio.subscribe( self.getName() );
        print( "INF: SoundReceiver: started!" );

        global clf, scaler, nummer, targetCoordinateList, effectorInit, motionProxy, postureProxy, tijd # , effectorName

        tijd = 0.0
        # lastCounter = 0
        filename1 = 'models/classifier1.pkl';
        filename2 = 'models/scaler.pkl';
        # nummer = 0
        clf = joblib.load(filename1);
        scaler = joblib.load(filename2);
        print( "INF: Classifier and scaler loaded!");

        # Init proxies.
        try:
            motionProxy = naoqi.ALProxy("ALMotion", self.strNaoIp, 9559)
        except Exception, e:
            print "Could not create proxy to ALMotion"
            print "Error was: ", e

        try:
            postureProxy = naoqi.ALProxy("ALRobotPosture", self.strNaoIp, 9559)
        except Exception, e:
            print "Could not create proxy to ALRobotPosture"
            print "Error was: ", e

        # Set NAO in Stiffness On, init movement
        StiffnessOn(motionProxy)

        space = motion.FRAME_ROBOT
        useSensor = False
        # coef = +1.0
        motionProxy.wakeUp()

        # Send NAO to Pose Init
        # postureProxy.goToPosture("StandInit", 0.5)
        postureProxy.goToPosture("Sit", 0.8)
        print( "INF: NAO ready")



        # self.processRemote( 4, 128, [18,0], [0]*128*4 );

    def stop( self ):
        print( "INF: SoundReceiver: stopping..." );
        audio = naoqi.ALProxy( "ALAudioDevice", self.strNaoIp, 9559 );
        audio.unsubscribe( self.getName() );
        print( "INF: SoundReceiver: stopped!" );
        if( self.outfile != None ):
            self.outfile.close();


    def processRemote( self, nbOfChannels, nbrOfSamplesByChannel, aTimeStamp, buffer ):
        print "hi"
        try:
            self.counter += 1
            # print audio.getFrontMicEnergy()
        except Exception, e:
            print "counterrrr", e
            print counter

        aSoundDataInterlaced = np.fromstring( str(buffer), dtype=np.int16 );

        print "Size of data:", aSoundDataInterlaced.shape, "\t\tGiven shape: (", nbOfChannels, "x", nbrOfSamplesByChannel, ")"
        reshape_size = aSoundDataInterlaced.shape[0] / nbOfChannels
        aSoundData = np.reshape( aSoundDataInterlaced, (nbOfChannels, reshape_size), 'F' );
        shape = aSoundData.shape
        #aSoundData = aSoundData[:,:nbrOfSamplesByChannel]
        print "size before:", shape, "\t\tshape after:", aSoundData.shape

        # print aSoundData.shape
        data11 = aSoundData[0,0:2400]
        data12 = aSoundData[0,2400:4800]
        data13 = aSoundData[0,4800:7200]

        data21 = aSoundData[1,0:2400]

        truncate_no = 680
        X11 = scipy.fftpack.rfft(data11)
        X12 = scipy.fftpack.rfft(data12)
        X13 = scipy.fftpack.rfft(data13)

        X21 = scipy.fftpack.rfft(data21)

        Xdb11 = 20*scipy.log10(scipy.absolute(X11))
        Xdb11 = Xdb11[0:truncate_no]
        Xdb11 = scaler.transform(Xdb11)

        Xdb12 = 20*scipy.log10(scipy.absolute(X12))
        Xdb12 = Xdb12[0:truncate_no]
        Xdb12 = scaler.transform(Xdb12)

        Xdb13 = 20*scipy.log10(scipy.absolute(X13))
        Xdb13 = Xdb13[0:truncate_no]
        Xdb13 = scaler.transform(Xdb13)

        Xdb21 = 20*scipy.log10(scipy.absolute(X21))
        Xdb21 = Xdb21[0:truncate_no]
        Xdb21 = scaler.transform(Xdb21)

        # averageV = np.mean( data1) # axis = 1
        y11 = clf.predict(Xdb11)
        y12 = clf.predict(Xdb12)
        y13 = clf.predict(Xdb13)

        y21 = clf.predict(Xdb21)
        energy = audio.getFrontMicEnergy()
        print energy

        if ((y11[0] + y12[0] + y13[0] + y21[0] > 0) and (self.lastCounter + 5 < self.counter) and (energy > 1500.0)): # and (tijd + 5.0 < time.time())): # y1 != [0] or y2 != [0] or y3 != [0]:
            print "signal get! rear mic says :", y21[0]
            try:
                y1, y2, y3 = [0], [0], [0]
                tijd = time.time()
                self.lastCounter = self.counter
                print "at", tijd, "energy:", energy
            except Exception, e:
                print e
            try:
                # fractionMaxSpeed = 0.7
                # axisMask         = 7 # just control position

                # motionProxy.setPosition(chainName, frame, target1, fractionMaxSpeed, axisMask)
                motionProxy.setAngles("LShoulderPitch", -1.2,0.7)
                # motionProxy.setAngles("RShoulderPitch", -3,0.8)
                time.sleep(5.0)
                # motionProxy.setPosition(chainName, frame, target2, fractionMaxSpeed, axisMask)
                # postureProxy.goToPosture("StandInit", 0.6)
                postureProxy.goToPosture("Sit", 0.8)
                print "moved"
                return None
                # time.sleep(4.0)
                # motionProxy.rest()
            except Exception, e:
                print "Could not move arm"
                print "Error was: ", e



    # processRemote - end

    def version( self ):
        return "0.6";

    # audio analysis, classification
    # split data into packets
    def split_data_samples(data):
        sample_size = 2400
        return zip(*[iter(data)]*sample_size)

    # normalize audio
    def normalize(snd_data):
        "Average the volume out"
        MAXIMUM = 16384
        times = float(MAXIMUM)/max(abs(i) for i in snd_data)

        r = array('h')
        for i in snd_data:
            r.append(int(i*times))
        return r

# get ffts
def fft_ana(data):
    X = scipy.fftpack.rfft(bit)
    Xdb = 20*scipy.log10(scipy.absolute(X))
    Xdb = Xdb[0:truncate_no]
    scaler.transform(Xdb)
    return Xdb


def StiffnessOn(proxy):
    # We use the "Body" name to signify the collection of all joints
    pNames = "Body"
    pStiffnessLists = 1.0
    pTimeLists = 1.0
    proxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def moveArm(proxy):
    print "ok!"
    targetCoordinate = [0.20, 0.12, 0.30]
    proxy.wbSetEffectorControl(effectorName, targetCoordinate)
    # for targetCoordinate in targetCoordinateList:
    #     targetCoordinate = [targetCoordinate[i] + effectorInit[i] for i in range(3)]
    #     proxy.wbSetEffectorControl("LArm", targetCoordinate)
    #     time.sleep(4.0)

def main():
    """ Main entry point

    """
    parser = OptionParser()
    parser.add_option("--pip",
        help="Parent broker port. The IP address or your robot",
        dest="pip")
    parser.add_option("--pport",
        help="Parent broker port. The port NAOqi is listening to",
        dest="pport",
        type="int")
    parser.set_defaults(
        pip=NAO_IP,
        pport=9559)

    (opts, args_) = parser.parse_args()
    pip   = opts.pip
    pport = opts.pport

    # We need this broker to be able to construct
    # NAOqi modules and subscribe to other modules
    # The broker must stay alive until the program exists
    myBroker = naoqi.ALBroker("myBroker",
       "0.0.0.0",   # listen to anyone
       0,           # find a free port and use it
       pip,         # parent broker IP
       pport)       # parent broker port


    # Warning: SoundReceiver must be a global variable
    # The name given to the constructor must be the name of the
    # variable
    global SoundReceiver
    SoundReceiver = SoundReceiverModule("SoundReceiver", pip)
    SoundReceiver.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print
        print "Interrupted by user, shutting down"
        myBroker.shutdown()
        sys.exit(0)



if __name__ == "__main__":
    main()
    #~ a = range(64);
    #~ print a
    #~ print np.reshape( a, (4, 16), 'F' );
