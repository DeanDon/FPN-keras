import cv2
import os
root_path='./UCF-101/'
write_path='/data/UCF-101-img/'
list_cls=os.listdir(root_path)
for cls in list_cls:
    write_cls_path=os.path.join(write_path,cls)
    if not os.path.exists(write_cls_path):
        os.mkdir(write_cls_path)
    cls_path=os.path.join(root_path,cls)
    list_video=os.listdir(cls_path)
    for video in list_video:
        print cls,video
        write_video_path=os.path.join(write_cls_path,video)
        if not os.path.exists(write_video_path):
            os.mkdir(write_video_path)
        video_path=os.path.join(cls_path,video)
        cap=cv2.VideoCapture(video_path)
        i=0
        _,img=cap.read(i)
        while(_):
            write_frame_path=os.path.join(write_video_path,str(i)+'.jpg')
            cv2.imwrite(write_frame_path,img)
            _, img = cap.read(i)
            i+=1
        cap.release()
