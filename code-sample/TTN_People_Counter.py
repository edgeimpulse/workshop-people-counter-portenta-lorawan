# Untitled - By: luisomoreau - Thu Aug 4 2022

# Edge Impulse - OpenMV Object Detection Example

import sensor, image, time, os, tf, math, uos, gc
from lora import *

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)

lora = Lora(band=BAND_EU868, poll_ms=60000, debug=False)       # Let the camera adjust.

print("Firmware:", lora.get_fw_version())
print("Device EUI:", lora.get_device_eui())
print("Data Rate:", lora.get_datarate())
print("Join Status:", lora.get_join_status())

appEui = "0000000000000001" # Add your App EUI here
appKey = "54F29F68BD8AE99603F750FD5BB1ED33" # Add your App Key here

net = None
labels = None
min_confidence = 0.5

peopleCounter = 0

try:
    # Load built in model
    labels, net = tf.load_builtin_model('trained')
except Exception as e:
    raise Exception(e)

try:
    lora.join_OTAA(appEui, appKey, timeout=20000)
    # Or ABP:
    #lora.join_ABP(devAddr, nwkSKey, appSKey, timeout=5000)
# You can catch individual errors like timeout, rx etc...
except LoraErrorTimeout as e:
    print("Something went wrong; are you indoor? Move near a window and retry")
    print("ErrorTimeout:", e)
except LoraErrorParam as e:
    print("ErrorParam:", e)

print("Connected.")
lora.set_port(3)

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

clock = time.clock()
now = time.ticks_ms()
while(True):
    clock.tick()

    img = sensor.snapshot()

    # detect() returns all objects found in the image (splitted out per class already)
    # we skip class index 0, as that is the background, and then draw circles of the center
    # of our objects

    for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
        if (i == 0): continue # background class
        if (len(detection_list) == 0): continue # no detections for this class?
        peopleCounter=len(detection_list)
        #print("********** %s **********" % labels[i])
        for d in detection_list:

            [x, y, w, h] = d.rect()
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            #print('x %d\ty %d' % (center_x, center_y))
            img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)

    #print(clock.fps(), "fps", end="\n\n")
    #print(peopleCounter, end="\n\n")

    # Send a message every minute
    uplink_interval = 60000

    if time.ticks_ms() - now > uplink_interval:
        print("Sending LoRa payload")
        try:
            # Uplink only
            lora.send_data(peopleCounter.to_bytes(1, 'big'), False)

            # Uplink + downlink request
            #if lora.send_data(peopleCounter.to_bytes(1, 'big'), True):
                #print("Message confirmed.")
            #else:
                #print("Message wasn't confirmed")

        except LoraErrorTimeout as e:
            print("ErrorTimeout:", e)

        # Reset the clock
        now = time.ticks_ms()


    # Read downlink messages
    # Not needed here
    #if (lora.available()):
        #data = lora.receive_data()
        #if data:
            #print("Port: " + data["port"])
            #print("Data: " + data["data"])
    #lora.poll()



