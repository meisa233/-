import av


# #container = av.open('/data/sv2007sbtest1/BG_34901.mpg')
# #container = av.open('/data/RAIDataset/Video/1.mp4')
# index = 0
# video = av.open('/data/sv2007sbtest1/BG_34901.mpg', 'r')
#
# container = av.open('/data/sv2007sbtest1/BG_34901.mpg')
# video = next(s for s in container.streams)
# for packet in container.demux(video):
#     a = packet.stream
#     for frame in packet.decode():
#         print "a"
#     # frame.to_image().save('/data/'+'frame-%04d.jpg' % frame.index)
#     # b = frame.index
#
#     print "a"

# container = av.open('/data/sv2007sbtest1/BG_34901.mpg')
#
# for frame in container.decode(video=0):
#     print frame

container = av.open('/data/sv2007sbtest1/BG_11362.mpg')
video = next(s for s in container.streams)
index = 0
for packet in container.demux():
    for frame in packet.decode():
        index += 1

print "a"
