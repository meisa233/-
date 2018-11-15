class JingweiXu():

    def get_vector(self, segments):
        import sys
        import os
        sys.path.insert(0, '/data/caffe/python')
        import caffe
        import cv2
        import numpy as np
        import math
        import csv

        caffe.set_mode_gpu()
        caffe.set_device(0)
        # load model(.prototxt) and weight (.caffemodel)

        # os.chdir('/data/Meisa/ResNet/ResNet-50')
        # ResNet_Weight = './resnet50_cvgj_iter_320000.caffemodel'  # pretrained on il 2012 and place 205

        os.chdir('/data/Meisa/hybridCNN')
        Hybrid_Weight = './hybridCNN_iter_700000.caffemodel'



        # ResNet_Def = 'deploynew_globalpool.prototxt'

        Hybrid_Def = 'Shot_hybridCNN_deploy_new.prototxt'
        net = caffe.Net(Hybrid_Def,
                        Hybrid_Weight,
                        caffe.TEST)

        # load video
        i_Video = cv2.VideoCapture('/data/RAIDataset/Video/1.mp4')

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        # get the number of frames of this video
        framenum = int(i_Video.get(7))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')




        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        net.blobs['data'].reshape(1,
                                  3,
                                  227, 227)

        FrameV = []

        if len(segments) == 1:
            i_Video.set(1, segments[0])
            ret, frame = i_Video.read()
            if frame is None:
                print i
            transformed_image = transformer.preprocess('data', frame)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            FrameV.extend(output['fc8'][0].tolist())
            #FrameV.extend(np.squeeze(output['global_pool'][0]).tolist())
            return FrameV

        for i in range(segments[0], segments[1]+1):
            i_Video.set(1, i)
            ret, frame = i_Video.read()
            if frame is None:
                print i
                continue
            transformed_image = transformer.preprocess('data', frame)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            FrameV.append(output['fc8'][0].tolist())
            # FrameV.append(np.squeeze(output['global_pool'][0]).tolist())

        return FrameV




    def RGBToGray(self, RGBImage):

        import numpy as np
        return np.dot(RGBImage[..., :3], [0.299, 0.587, 0.114])



    def getHist(self, segments):
        import cv2
        i_Video = cv2.VideoCapture('/data/RAIDataset/Video/1.mp4')
        i_Video.set(1, segments[0])
        ret1, frame1 = i_Video.read()

        i_Video.set(1, segments[1])
        ret2, frame2 = i_Video.read()

        frame1hist = cv2.calcHist(frame1,)



    def CutVideoIntoSegments(self):
        import math
        import cv2
        import numpy as np

        # It save the pixel intensity between 20n and 20(n+1)
        d = []
        SegmentsLength = 21
        i_Video = cv2.VideoCapture('/data/RAIDataset/Video/1.mp4')
        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / float(SegmentsLength)))
        for i in range(Count):

            i_Video.set(1, (SegmentsLength-1)*i)
            ret1, frame_20i = i_Video.read()

            if((SegmentsLength-1)*(i+1)) >= FrameNumber:
                i_Video.set(1, FrameNumber-1)
                ret2, frame_20i1 = i_Video.read()
                d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))
                break

            i_Video.set(1, (SegmentsLength-1)*(i+1))
            ret2, frame_20i1 = i_Video.read()

            d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))


        # The number of group
        GroupNumber = int(math.ceil(float(FrameNumber) / 10.0))

        MIUG = np.mean(d)
        a = 0.7 # The range of a is 0.5~0.7
        Tl = [] # It save the Tl of each group
        CandidateSegment = []
        for i in range(GroupNumber):
            MIUL = np.mean(d[10*i:10*i+10])
            SigmaL = np.std(d[10*i:10*i+10])

            Tl.append(MIUL + a*(1+math.log(MIUG/MIUL))*SigmaL)
            for j in range(10):
                if i*10 + j >= len(d):
                    break
                if d[i*10+j]>Tl[i]:
                    CandidateSegment.append([(i*10+j)*(SegmentsLength-1), (i*10+j+1)*(SegmentsLength-1)])
                    #print 'A candidate segment is', (i*10+j)*20, '~', (i*10+j+1)*20

        return CandidateSegment
        #print 'a'



    # Calculate the cosin distance between vector1 and vector2
    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    # Calculate the D1
    def getD1(self, Segment):
        return self.cosin_distance(Segment[0], Segment[-1])



    # Check the segments selected (by the function called CutVideoIntoSegments) whether have cut
    def CheckSegments(self, CandidateSegments):
        GroundTruth = []
        with open('/data/RAIDataset/Video/gt_1.txt', 'r') as f:
            GroundTruth = f.read()

        for i in range(len(GroundTruth)):
            for j in range(len(CandidateSegments)):
                if GroundTruth[i][1] >= CandidateSegments[j][0] && GroundTruth[i+1][0] <= CandidateSegments[j][0]:
                    break
                elif GroundTruth[i][1] < CandidateSegments[j][0]:
                    print 'This cut "', GroundTruth[i][1],',', GroundTruth[i+1][0],'"can not be detected'
    # CT Detection
    def CTDetection(self):
        import matplotlib.pyplot as plt
        import numpy as np

        k = 0.4
        Tc = 0.55

        CandidateSegments = self.CutVideoIntoSegments()
        # for i in range(len(CandidateSegments)):
        #     FrameV = self.get_vector(CandidateSegments[i])

        for i in range(len(CandidateSegments)):
            FrameV = []
            FrameV.append(self.get_vector([CandidateSegments[i][0]]))
            FrameV.append(self.get_vector([CandidateSegments[i][-1]]))

            D1 = self.getD1(FrameV)
            if D1 < 0.9:
                D1Sequence = []

                CandidateFrame = self.get_vector(CandidateSegments[i])
                for j in range(len(CandidateFrame) - 1):
                    D1Sequence.append(self.cosin_distance(CandidateFrame[j], CandidateFrame[j+1]))
                if np.min(D1Sequence) < k*D1+(1-k):
                    if np.max(D1Sequence)- np.min(D1Sequence) > Tc:
                        print np.argmin(D1Sequence)

            # # plot the image
            #
            # x = range(len(D1Sequence))
            #
            # plt.figure()
            # plt.plot(x, D1Sequence)
            #
            # plt.show()


if __name__ == '__main__':
    test1 = JingweiXu()
    #test1.CutVideoIntoSegments()
    test1.CTDetection()
