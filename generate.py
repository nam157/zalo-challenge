from facenet_pytorch import MTCNN
import cv2
import os,glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

save_dir = 'G:/zalo_challenge/liveness_face/datasets/datasets_train/'
os.makedirs(save_dir,exist_ok=True)
model = MTCNN()
skip_num = 3
file_list = open(save_dir+"/file_list.txt","w")
for file in glob.glob("G:/zalo_challenge/liveness_face/datasets/train/videos/*.mp4"):
    print("Processing video %s"%file)
    # dir_name = os.path.join(save_dir, *file.replace(".mp4", "").split("/")[-3:])
    
    vidcap = cv2.VideoCapture(file)
    success, frame = vidcap.read()
    count = 0
    frame_num = 0

    while success:
        detect_res = model.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if len(detect_res)>0 and count%skip_num==0 :
            file_name = os.path.join(save_dir,f"{os.path.basename(file)[:-4]}_frame_{frame_num}.jpg")
            bbox = detect_res[0]
            # cv2.rectangle(frame,pt1=(int(bbox[0][0]),int(bbox[0][1])),pt2=(int(bbox[0][2]),int(bbox[0][3])),thickness=2,color=(0,244,244))
            label_org = pd.read_csv('G:/zalo_challenge/liveness_face/datasets/train/label.csv')
            for i in range(len(label_org)):
                if os.path.basename(file) == label_org.loc[i,'fname']:
                    label = label_org.loc[i,'liveness_score']
            
            try:
                print(file_name,int(bbox[0][0]),int(bbox[0][1]),int(bbox[0][2]),int(bbox[0][3]),label)
                file_list.writelines("%s %d %d %d %d %d\n"%(file_name,int(bbox[0][0]),int(bbox[0][1]),int(bbox[0][2]),int(bbox[0][3]),label))
            except:
                break
            cv2.imwrite(file_name,frame)
            frame_num +=1
        success, frame = vidcap.read()
        count +=1
        # cv2.imshow('test',frame)
        # cv2.waitKey(1)
    vidcap.release()