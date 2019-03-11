class New_Method():

    def GetPixDiff(self, frame1, frame2, allpixels):
        import numpy as np
        return np.sum(np.power(np.abs(frame1-frame2), 2))/float(allpixels)

    # Get the Color Histogram from "frame"
    def GetFrameHist(self, frame, binsnumber):
        import cv2
        Bframehist = cv2.calcHist([frame], channels=[0], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        Gframehist = cv2.calcHist([frame], channels=[1], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        Rframehist = cv2.calcHist([frame], channels=[2], mask=None, ranges=[0.0, 255.0], histSize=[binsnumber])
        return [Bframehist, Gframehist, Rframehist]

    # Get the Manhattan Distance
    def Manhattan(self, vector1, vector2):
        import numpy as np
        return np.sum(np.abs(vector1 - vector2))

    # Get the Manhattan distance between the histogram of frame1 and frame2
    def getHist_Manhattan(self, frame1, frame2, allpixels):

        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        distance_Manhattan = self.Manhattan(Bframe1hist, Bframe2hist) + self.Manhattan(Gframe1hist,
                                                                                       Gframe2hist) + self.Manhattan(
            Rframe1hist, Rframe2hist)
        return distance_Manhattan / allpixels

    # Get the chi square distance between the histogram of frame1 and frame2
    def getHist_chi_square(self, frame1, frame2, allpixels):
        import cv2
        binsnumber = 64

        [Bframe1hist, Gframe1hist, Rframe1hist] = self.GetFrameHist(frame1, binsnumber)
        [Bframe2hist, Gframe2hist, Rframe2hist] = self.GetFrameHist(frame2, binsnumber)

        chi_square_distance = cv2.compareHist(Bframe1hist, Bframe2hist, method=cv2.HISTCMP_CHISQR) + cv2.compareHist(
            Gframe1hist, Gframe2hist, method=cv2.HISTCMP_CHISQR) + cv2.compareHist(Rframe1hist, Rframe2hist,
                                                                                   method=cv2.HISTCMP_CHISQR)
        return chi_square_distance / (allpixels)


    def if_overlap(self, begin1, end1, begin2, end2):
        if begin1 > begin2:
            begin1, end1, begin2, end2 = begin2, end2, begin1, end1

        return end1 >= begin2

    def DetectHardCut(self,VideoPath, NewCandidateSegments):
        import cv2
        import numpy as np

        i_Video = cv2.VideoCapture(VideoPath)
        # get width of this video
        wid = int(i_Video.get(3))
        # get height of this video
        hei = int(i_Video.get(4))
        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        AllPixels = wid * hei

        FrameFirst = []

        HardCut = []
        for i in range(len(NewCandidateSegments)):
            d = []
            i_Video.set(1, NewCandidateSegments[i][0])
            ret1, FrameFirst = i_Video.read()
            for j in range(NewCandidateSegments[i][0]+1 ,  NewCandidateSegments[i][1]+1):
                i_Video.set(1, j)
                ret2, FrameNext = i_Video.read()
                d.append(0.5 * self.getHist_chi_square(FrameFirst, FrameNext, AllPixels) + 0.5 * self.GetPixDiff(
                    cv2.cvtColor(FrameFirst, cv2.COLOR_BGR2GRAY), cv2.cvtColor(FrameNext, cv2.COLOR_BGR2GRAY),
                    AllPixels))
                FrameFirst = FrameNext
            if np.sum(d) - np.min(d) > 40 and len([k for k in d if k>40])==1:
                HardCut.append([NewCandidateSegments[i][0]+np.argmax(d), NewCandidateSegments[i][0]+np.argmax(d)+1])
        print "a"

    def GetCandidateSegments(self, VideoPath, HardTruth, GraTruth):
        import cv2
        import math
        import copy

        HardCut_SegmentsLength1 = 10
        HardCut_SegmentsLength2 = 5

        i_Video = cv2.VideoCapture(VideoPath)

        # get width of this video
        wid = int(i_Video.get(3))
        # get height of this video
        hei = int(i_Video.get(4))
        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        AllPixels = wid * hei

        Count = int(math.ceil(FrameNumber/10.0))

        Frames10_1 = []
        Frames10_2 = []
        d10 = []

        Frames10_5_1 = []
        Frames10_5_2 = []
        d10_5 = []

        i_Video.set(1,0)
        ret1, Frames10_1 = i_Video.read()
        CandidateSegments1 = []
        CandidateSegments2 = []
        for i in range(1, Count):
            # print i*HardCut_SegmentsLength1
            i_Video.set(1, i*HardCut_SegmentsLength1)
            ret2, Frames10_2 = i_Video.read()
            d = self.getHist_chi_square(Frames10_2, Frames10_1, AllPixels)
            if d > 0.5:
                CandidateSegments1.append([(i-1)*HardCut_SegmentsLength1, i*HardCut_SegmentsLength1])
                d10.append(d)
            Frames10_1 = copy.deepcopy(Frames10_2)


        if FrameNumber%10==FrameNumber%5:
            Count1 = Count - 1
        else:
            Count1 = Count

        i_Video.set(1, 5)
        ret1, Frames10_5_1 = i_Video.read()


        for i in range(1, Count1):
            if i * HardCut_SegmentsLength1 + HardCut_SegmentsLength2>=FrameNumber:
                break
            i_Video.set(1, i * HardCut_SegmentsLength1 + HardCut_SegmentsLength2)
            ret2, Frames10_5_2 = i_Video.read()
            d = self.getHist_chi_square(Frames10_5_1, Frames10_5_2, AllPixels)
            if d > 0.5:
                CandidateSegments2.append([(i-1)*HardCut_SegmentsLength1+HardCut_SegmentsLength2, i*HardCut_SegmentsLength1+HardCut_SegmentsLength2])
                d10_5.append(d)
            Frames10_5_1 = copy.deepcopy(Frames10_5_2)

        NewCandidateSegments = []

        # for i in range(len(CandidateSegments1)):
        #     for j in range(len(CandidateSegments2)):
        #         if CandidateSegments1[i][0] > CandidateSegments2[j][1]:
        #             continue
        #         elif CandidateSegments1[i][1] - CandidateSegments2[j][0]==5:
        #             if d10[i]>d10_5[j]:
        #                 NewCandidateSegments.append(CandidateSegments1[i])
        #             else:
        #                 NewCandidateSegments.append(CandidateSegments2[j])
        #             break
        #         elif CandidateSegments1[i][0] - CandidateSegments2[j][1] == -5:
        #             if d10[i]>d10_5[j]:
        #                 NewCandidateSegments.append(CandidateSegments1[i])
        #             else:
        #                 NewCandidateSegments.append(CandidateSegments2[j])
        #         elif CandidateSegments1[i][1] < CandidateSegments2[j][0]:
        #             break

        i = 0
        j = 0
        while i < len(CandidateSegments1) or j < len(CandidateSegments2):
            if CandidateSegments1[i][1] < CandidateSegments2[j][0]:
                NewCandidateSegments.append(CandidateSegments1[i])
                if i < len(CandidateSegments1):
                    i += 1
            elif CandidateSegments1[i][0] > CandidateSegments2[j][1]:
                NewCandidateSegments.append(CandidateSegments2[j])
                if j < len(CandidateSegments2):
                    j += 1
            elif CandidateSegments1[i][0] < CandidateSegments2[j][1] and CandidateSegments1[i][1] > CandidateSegments2[j][1]:
                if d10[i] > d10_5[j]:
                    NewCandidateSegments.append(CandidateSegments1[i])
                else:
                    NewCandidateSegments.append(CandidateSegments2[j])
                if i < len(CandidateSegments1):
                    i += 1
                if j < len(CandidateSegments2):
                    j += 1
            elif CandidateSegments1[i][1] > CandidateSegments2[j][0] and CandidateSegments2[j][1] > CandidateSegments1[i][1]:
                if d10[i] > d10_5[j]:
                    NewCandidateSegments.append(CandidateSegments1[i])
                else:
                    NewCandidateSegments.append(CandidateSegments2[j])
                if i < len(CandidateSegments1):
                    i += 1
                if j < len(CandidateSegments2):
                    j += 1
            if i == len(CandidateSegments1) and j < len(CandidateSegments2):
                NewCandidateSegments.extend(CandidateSegments2[j:])
                break
            elif j == len(CandidateSegments2) and i < len(CandidateSegments1):
                NewCandidateSegments.extend(CandidateSegments1[i:])
                break

        return NewCandidateSegments



    def GetLabels(self, LabelTXT):
        HardTruth = []
        GraTruth = []
        with open(LabelTXT) as f:
            AllLines = f.readlines()
        GroundTruth = [[int(AllLines[0].strip().split('\t')[-1])]]
        for i in range(1, len(AllLines)-1):
            GroundTruth[-1].extend([int(AllLines[i].strip().split('\t')[0])])
            GroundTruth.append([int(AllLines[i].strip().split('\t')[1])])
        GroundTruth[-1].extend([int(AllLines[-1].strip().split('\t')[0])])

        for i in GroundTruth:
            if i[1]-i[0] == 1:
                HardTruth.append(i)
            else:
                GraTruth.append(i)

        return [HardTruth, GraTruth]


    def DetectionOnRAIDataset(self):
        from glob import glob

        LabelFilePath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\gt_'
        VideoPath = 'E:\\Meisa_SiameseNetwork\\RAIDataset\\videos\\'

        AllHard = 0
        AllMiss = 0

        VideosNumber = 10

        for i in range(6, VideosNumber+1):
            [HardTruth, GraTruth] = self.GetLabels(LabelFilePath + str(i) + '.txt')
            self.DetectHardCut(VideoPath + str(i) + '.mp4', self.GetCandidateSegments(VideoPath + str(i) + '.mp4', HardTruth, GraTruth))

if __name__ == '__main__':
    test1 = New_Method()
    test1.DetectionOnRAIDataset()
