class JingweiXu():



    # Get the Manhattan Distance between vector1 and vector2
    def Manhattan(self, vector1, vector2):
        import numpy as np
        return np.sum(np.abs(vector1 - vector2))

    # Get the Color Histogram from "frame"
    def GetFrameHist(self, frame, binsnumber):
        import cv2
        Bframehist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        Gframehist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        Rframehist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0,255.0], histSize=[binsnumber])
        return [Bframehist, Gframehist, Rframehist]

    # Get the Manhattan distance between the histogram of frame1 and frame2
    def getHist_Manhattan(self, frame1, frame2, allpixels):

        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        distance_Manhattan = self.Manhattan(Bframe1hist, Bframe2hist) + self.Manhattan(Gframe1hist, Gframe2hist) + self.Manhattan(Rframe1hist, Rframe2hist)
        return distance_Manhattan/allpixels

    # Get the chi square distance between the histogram of frame1 and frame2
    def getHist_chi_square(self, frame1, frame2, allpixels):
        import cv2
        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        chi_square_distance = cv2.compareHist(Bframe1hist, Bframe2hist, method=cv2.HISTCMP_CHISQR)+cv2.compareHist(Gframe1hist, Gframe2hist, method=cv2.HISTCMP_CHISQR)+cv2.compareHist(Rframe1hist, Rframe2hist, method=cv2.HISTCMP_CHISQR)
        return chi_square_distance/(allpixels)

    # Cut the video into candidate segments(every segment's length is 10)
    def CutVideoIntoSegmentsBaseOnNeuralNet(self, VideoPath):
        import math
        import cv2
        import numpy as np
        import caffe

        caffe.set_mode_gpu()
        caffe.set_device(0)

        SqueezeNet_Def = '/data/SqueezeNet/deploy.prototxt'
        SqueezeNet_Weight = '/data/SqueezeNet/squeezenet_v1.1.caffemodel'
        net = caffe.Net(SqueezeNet_Def,
                        SqueezeNet_Weight,
                        caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        net.blobs['data'].reshape(1,
                                  3,
                                  227, 227)


        # It save the pixel intensity between 20n and 20(n+1)
        d = []

        SegmentsLength = 11

        i_Video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        LastFrameIndex = FrameNumber - 1

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / float(SegmentsLength-1)))
        for i in range(Count):

            frame1add = 0
            frame2add = 0

            i_Video.set(1, (SegmentsLength-1)*i)
            ret1, frame_20i = i_Video.read()

            while frame_20i is None:
                frame1add += 1
                i_Video.set(1, (SegmentsLength-1)*i+frame1add)
                ret1, frame_20i = i_Video.read()

            if((SegmentsLength-1)*(i+1)) >= FrameNumber:
                i_Video.set(1, FrameNumber-1)
                ret2, frame_20i1 = i_Video.read()
                # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))

                while frame_20i1 is None:
                    frame2add += 1
                    i_Video.set(1, FrameNumber - 1 - frame2add)
                    LastFrameIndex = FrameNumber -1 - frame2add
                    ret2, frame_20i1 = i_Video.read()

                transformed_image = transformer.preprocess('data', frame_20i1)
                net.blobs['data'].data[...] = transformed_image
                output = net.forward()
                Frame_Last = np.squeeze(output['pool10'][0]).tolist()


                transformed_image = transformer.preprocess('data', frame_20i)
                net.blobs['data'].data[...] = transformed_image
                output = net.forward()
                Frame_First = np.squeeze(output['pool10'][0]).tolist()

                d.append(self.cosin_distance(Frame_First, Frame_Last))
                # d.append(self.getHist(frame_20i, frame_20i1, wid*hei))
                break

            i_Video.set(1, (SegmentsLength-1)*(i+1))
            ret2, frame_20i1 = i_Video.read()

            # d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))
            transformed_image = transformer.preprocess('data', frame_20i1)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            Frame_Last = np.squeeze(output['pool10'][0]).tolist()


            transformed_image = transformer.preprocess('data', frame_20i)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            Frame_First = np.squeeze(output['pool10'][0]).tolist()

            d.append(self.cosin_distance(Frame_First, Frame_Last))


        GroupLength = 10
        # The number of group
        GroupNumber = int(math.ceil(float(len(d)) / GroupLength))

        MIUG = np.mean(d)
        a = 0.7 # The range of a is 0.5~0.7
        Tl = [] # It save the Tl of each group
        CandidateSegment = []
        for i in range(GroupNumber):

            MIUL = np.mean(d[GroupLength*i:GroupLength*i+GroupLength])
            SigmaL = np.std(d[GroupLength*i:GroupLength*i+GroupLength])

            Tl.append(MIUL + a*(1+math.log(MIUG/MIUL))*SigmaL)
            for j in range(GroupLength):
                if i*GroupLength + j >= len(d):
                    break
                if d[i*GroupLength+j]<Tl[i]:
                    CandidateSegment.append([(i*GroupLength+j)*(SegmentsLength-1), (i*GroupLength+j+1)*(SegmentsLength-1)])
                    #print 'A candidate segment is', (i*10+j)*20, '~', (i*10+j+1)*20


        for i in range(1,len(d)-1):
            if (d[i]>(3*d[i-1]) or d[i]>(3*d[i+1])) and d[i]> 0.8 * MIUG:
                if [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)] not in CandidateSegment:
                    j = 0
                    while j < len(CandidateSegment):
                        if (i+1)*(SegmentsLength-1)<= CandidateSegment[j][0]:
                            CandidateSegment.insert(j, [i*(SegmentsLength-1), (i+1)*(SegmentsLength-1)])
                            break
                        j += 1

        if CandidateSegment[-1][1]>LastFrameIndex:
            CandidateSegment[-1][1] = LastFrameIndex
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


####################################The Following is used for evaluating################################################
    def if_overlap(self, begin1, end1, begin2, end2):
        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2


    def get_union_cnt(self,set1, set2):
        cnt = 0
        for begin, end in set1:
            for _begin, _end in set2:
                if self.if_overlap(begin, end, _begin, _end):
                    cnt += 1
                    break
        return cnt

    def recall_pre_f1(self,a, b, c):
        a = float(a)
        b = float(b)
        c = float(c)
        recall = a / b if b != 0 else 0
        precison = a / c if c != 0 else 0
        f1 = 2 * recall * precison / (recall + precison)
        return precison, recall, f1

    def eval(self, predict, gt):


        gt_cuts = [(begin,end) for begin,end in gt if end-begin==1]
        gt_graduals = [(begin, end) for begin, end in gt if end - begin > 1]

        predicts_cut = [(begin,end) for begin,end in predict if end-begin==1]
        predicts_gradual = [(begin, end) for begin, end in predict if end - begin > 1]

        cut_correct = self.get_union_cnt(gt_cuts, predicts_cut)
        gradual_correct = self.get_union_cnt(gt_graduals, predicts_gradual)
        all_correct = self.get_union_cnt(predicts_cut + predicts_gradual, gt)

        return [cut_correct, gradual_correct, all_correct]

    ##################################################################################



    # Check the segments selected (by the function called CutVideoIntoSegments) whether have cut
    def CheckSegments(self, CandidateSegments, HardCutTruth, GradualTruth):


        import numpy as np
        MissHard = []
        MissGra = []
        for i in range(len(HardCutTruth)):
            for j in range(len(CandidateSegments)):
                if CandidateSegments[j][1] < HardCutTruth[i][0]:
                    continue
                if self.if_overlap(CandidateSegments[j][0], CandidateSegments[j][1], HardCutTruth[i][0], HardCutTruth[i][1]):
                    break
                if CandidateSegments[j][0] > HardCutTruth[i][1]:
                    MissHard.append(HardCutTruth[i])
                    break

        for i in range(len(GradualTruth)):
            for j in range(len(CandidateSegments)):
                if CandidateSegments[j][1] < GradualTruth[i][0]:
                    continue
                if self.if_overlap(CandidateSegments[j][0], CandidateSegments[j][1], GradualTruth[i][0], GradualTruth[i][1]):
                    break
                if CandidateSegments[j][0] > GradualTruth[i][1]:
                    MissGra.append(GradualTruth[i])
                    break

        print "MissHard No. is ", len(MissHard)
        print "MissGra No. is ", len(MissGra)


        print 'Hard Rate is ', (len(HardCutTruth) - len(MissHard))/float(len(HardCutTruth))
        print 'Gra Rate is ', (len(GradualTruth) - len(MissGra))/float(len(GradualTruth))

        return [HardCutTruth, GradualTruth]





    def CTDetectionBaseOnHist(self, VideoPath, HardCutTruth, GradualTruth):
        import numpy as np
        import cv2
        import math


        k = 0.4
        Tc = 0.05

        # CandidateSegments = self.CutVideoIntoSegments()

        CandidateSegments = self.CutVideoIntoSegmentsBaseOnNeuralNet(VideoPath)

        self.CheckSegments(CandidateSegments, HardCutTruth, GradualTruth)

        # It saves the predicted shot boundaries
        Answer = []

        # It saves the candidate segments which may have gradual
        CandidateGra = []

        i_Video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        # get the number of frames of this video
        FrameNum = int(i_Video.get(7))

        # It saves the predicted transition numbers
        AnswerLength = 0


        for i in range(len(CandidateSegments)):
            frame1add = 0
            frame2add = 0
            # frame1 saves the first frame of the segment's
            i_Video.set(1, CandidateSegments[i][0])
            ret1, frame1 = i_Video.read()

            # Consider the situation that the frame that would be not extracted
            while frame1 is None:
                frame1add += 1
                i_Video.set(1, CandidateSegments[i][0]+frame1add)
                ret1, frame1 = i_Video.read()

            # frame2 saves the last frame of the segment's
            i_Video.set(1, CandidateSegments[i][1])
            ret1, frame2 = i_Video.read()

            # Consider the situation that the frame that would be not extracted
            while frame2 is None:
                frame2add +=1
                i_Video.set(1, CandidateSegments[i][1]-frame2add)
                ret1, frame2 = i_Video.read()

            HistDifference = []

            # if CandidateSegments[i][0]>=14130:
                # print 'a'
            if self.getHist_Manhattan(frame1, frame2, wid * hei) >= 0.45:
                # Calculate the Manhattan distance from the frame1 and frame2 (Hist)
                for j in range(CandidateSegments[i][0], CandidateSegments[i][1]):
                    jadd1 = 0
                    jadd2 = 0
                    i_Video.set(1, j)
                    ret1_, frame1_ = i_Video.read()

                    i_Video.set(1, j+1)
                    ret2_, frame2_ = i_Video.read()

                    HistDifference.append(self.getHist_chi_square(frame1_, frame2_, wid*hei))


                if np.max(HistDifference) > 0.1:# and len([_ for _ in HistDifference if _>0.1])<len(HistDifference):
                    CandidatePeak = -1
                    MAXValue = -1

                    # Spectial Situation #1
                    if HistDifference[0] > 0.1 and HistDifference[0] > HistDifference[1]:
                        CandidatePeak = 0
                        MAXValue = HistDifference[0] - HistDifference[1]

                    for ii in range(1,len(HistDifference)-1):
                        if HistDifference[ii]>0.1 and HistDifference[ii] > HistDifference[ii-1] and HistDifference[ii] > HistDifference[ii+1]:
                            if np.max([np.abs(HistDifference[ii]-HistDifference[ii-1]), np.abs(HistDifference[ii]-HistDifference[ii+1])])>MAXValue:
                                CandidatePeak = ii
                                MAXValue = np.max([np.abs(HistDifference[ii]-HistDifference[ii-1]), np.abs(HistDifference[ii]-HistDifference[ii+1])])

                    if HistDifference[-1] > 0.1 and HistDifference[-1] > HistDifference[-2] and (HistDifference[-1]-HistDifference[-2])>MAXValue:
                        CandidatePeak = len(HistDifference)-1
                        MAXValue = HistDifference[-1]-HistDifference[-2]
                    if MAXValue>-1:
                        Answer.append(([CandidateSegments[i][0]+CandidatePeak, CandidateSegments[i][0]+CandidatePeak+1]))
                        # if MAXValue>20 and len([_ for _ in HistDifference if _<0.1])==len(HistDifference)-1 and (np.argmax(HistDifference)!=0 and np.argmax(HistDifference)!=len(HistDifference)-1):
                        #     AbsoluteCut.append([CandidateSegments[i][0]+CandidatePeak, CandidateSegments[i][0]+CandidatePeak+1])
                                # print a
                else:
                    for k1 in HardCutTruth:
                        if self.if_overlap(CandidateSegments[i][0], CandidateSegments[i][1], k1[0], k1[1]) and \
                                Answer[-1] != k1:
                            print "cut", k1, "missed"

            else:
                for k2 in HardCutTruth:
                    if self.if_overlap(CandidateSegments[i][0], CandidateSegments[i][1], k2[0], k2[1]) and len(Answer)>0 and Answer[-1]!=k2:
                        print 'This cut has been missed : ', k2

            #     Answer.append([CandidateSegments[i][0]+np.argmax(HistDifference), CandidateSegments[i][0]+np.argmax(HistDifference)+1])
            # elif np.max(HistDifference) > 0.5 and len([_ for _ in HistDifference if _ >0.5]) == 1 and (np.max(HistDifference)/np.max([_ for _ in HistDifference if _ <=0.5]))>=10 :
            #     Answer.append([CandidateSegments[i][0]+np.argmax(HistDifference), CandidateSegments[i][0]+np.argmax(HistDifference)+1])
            # elif np.max(HistDifference) > 0.5 and len([_ for _ in HistDifference if _ >0.5]) == 2 and (np.max(HistDifference)/np.min([_ for _ in HistDifference if _ >0.5])) >10:
            #     Answer.append([CandidateSegments[i][0]+np.argmax(HistDifference), CandidateSegments[i][0]+np.argmax(HistDifference)+1])

                # if Answer[-1] == [1589, 1590]:
                #     print 'a'
            if len(Answer) > 0 and len(Answer) > AnswerLength:
                AnswerLength += 1
                if Answer[-1] not in HardCutTruth:
                    print 'This a false cut'
                # Flag = False
                # for k in HardCutTruth:
                #     Flag = self.if_overlap(Answer[-1][0], Answer[-1][1], k[0], k[1])
                #     if Flag:
                #         break
                # if Flag is False:
                #     print 'This is a false cut: ', Answer[-1]



        Miss = 0
        True_ = 0
        False_ = 0

        # AbsoluteFalse = 0
        for i in Answer:
            if i not in HardCutTruth:
                print 'False :', i, '\n'
                False_ = False_ + 1
            else:
                True_ = True_ + 1




        for i in HardCutTruth:
            if i not in Answer:
                Miss = Miss + 1

        print 'False No. is', False_,'\n'
        print 'True No. is', True_, '\n'
        print 'Miss No. is', Miss, '\n'
        # print 'The false(MaxValue>20) No. is', AbsoluteFalse


    def GetLabels(self, xmlfile):

        HardTruth = []
        GraTruth = []
        for i in range(len(xmlfile)):
            if 'CUT' in xmlfile[i]:
                HardTruth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])
            elif 'DIS' in xmlfile[i] or 'OTH' in xmlfile[i]:
                GraTruth.append([int(xmlfile[i].split('"')[-4]), int(xmlfile[i].split('"')[-2])])

        return [HardTruth, GraTruth]

    def TRECVidSBD(self):
        import os
        from glob import glob

        VideoPath = '/data/sv2007sbtest1/'

        # The path of TRECVid2007
        os.chdir('/data/sv2007sbtest1')

        # Get all .mpg videos
        AllMpg = glob('./*.mpg')

        Allxml = glob('./sbref071/ref/ref*.xml')

        for i in range(len(AllMpg)):
            with open(Allxml[i]) as f:
                xmlfile = f.readlines()
            [HardTruth, GraTruth] = self.GetLabels(xmlfile)
            self.CTDetectionBaseOnHist(VideoPath + AllMpg[i].split('/')[-1], HardTruth, GraTruth)

        print 'a'

if __name__ == '__main__':
    test1 = JingweiXu()
    # test1.CTDetection()
    # test1.CutVideoIntoSegments()

    # test1.CTDetectionBaseOnHist()
    # test1.CheckSegments(test1.CutVideoIntoSegmentsBaseOnNeuralNet())

    test1.TRECVidSBD()