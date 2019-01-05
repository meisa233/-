from glob import glob
import cv2
import os
# Extract the frame from videos from UCF-101 dataset and convert them to images
# The name of converted images are "%06d.jpg" and notice that the name of image into C3D (in caffe) from 1 !!
if __name__ == '__main__':
    AllFolders = glob('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/UCF-101/*')
    # print AllFolders
    # os.chdir('/data/C3D/C3D-v1.1')
    for i in AllFolders:
        VideoPath = [i, "*.avi"]
        AllVideos = glob("/".join(VideoPath))
        os.mkdir('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/%s' % i.split('/')[-1])
        for j in AllVideos:
            os.chdir('/data/C3D/C3D-v1.1/data/users/trandu/datasets/ucf101/frm/%s' % i.split('/')[-1])
            i_Video = cv2.VideoCapture(j)
            success = True
            framenumber = 1
            os.mkdir(j.split('/')[-1].split('.')[0])
            os.chdir(j.split('/')[-1].split('.')[0])
            while(success):
                success, frame = i_Video.read()
                cv2.imwrite('%06d.jpg' % framenumber, frame)
                framenumber = framenumber + 1
