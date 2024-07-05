import cv2
import os

def video2frame(videos_path, frames_save_path, time_interval):
    # 确保保存路径存在
    os.makedirs(frames_save_path, exist_ok=True)
    
    cap = cv2.VideoCapture(videos_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {videos_path}")
        return

    count = 0
    success = True

    while success:
        success, image = cap.read()

        if not success:
            print(f"Error: Could not read frame {count} from video.")
            break

        if image is None or image.size == 0:
            print(f"Warning: Skipping empty frame {count}.")
            continue

        if count % time_interval == 0:
            frame_path = os.path.join(frames_save_path, f"frame{count}.jpg")
            if not cv2.imencode('.jpg', image)[1].tofile(frame_path):
                print(f"Error: Could not save frame {count} to {frame_path}")
        
        count += 1
    
    cap.release()
    print(f"Finished extracting frames. Total frames: {count}")
  
if __name__ == '__main__':
   videos_path = '/data/cv/jiahe.lu/nniefacelib/PFPLD/landmark10181138.avi'
   video_name = videos_path.split('/')[-1][:-4]
   frames_save_path = f'/data/cv/jiahe.lu/nniefacelib/PFPLD/test_related/video_frame/{video_name}'
#    os.makedirs(frames_save_path)
   time_interval = 2   # 隔一帧保存一次
   video2frame(videos_path, frames_save_path, time_interval)