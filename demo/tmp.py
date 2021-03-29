import os
import xml.etree.ElementTree as ET


count1 = 0
count2 = 0
partitions = os.listdir("/gruntdata4/dzr/ILSVRC2015/Data/VID/train/")
for partition in partitions:
    videos = os.listdir(os.path.join("/gruntdata4/dzr/ILSVRC2015/Data/VID/train/", partition))
    for video in videos:
        xmls = os.listdir(os.path.join("/gruntdata4/dzr/ILSVRC2015/Annotations/VID/train/",partition, video))
        JPEGs = os.listdir(os.path.join("/gruntdata4/dzr/ILSVRC2015/Data/VID/train/", partition, video))
        if len(xmls) != len(JPEGs):
            print(len(xmls), len(JPEGs))
        for xml in xmls:
            tree = ET.parse(os.path.join("/gruntdata4/dzr/ILSVRC2015/Annotations/VID/train/", partition, video, xml))
            if len(tree.findall("object")) == 0:
                count1 += 1
            else:
                count2 += 1

"""
>>> count1
36265
>>> count2
1086132
>>> count1 / count2
0.033389127656675247
"""
