import cv2
import mediapipe as mp
import os
import csv , json
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192,192,192) #grey
header = ['img_name', 'NOSE', 'LEFT_EYE', 'RIGHT_EYE' , 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
if not os.path.exists(os.getcwd()+"\datasheet2.csv"):
    with open(os.getcwd()+"\datasheet2.csv", 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy() #annotation means tagging (human interference)
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

path = "C:/Users/sathish sp/Desktop/py.programs/Welding/Stacking/"
dir_list = os.listdir(path)
# dir_list = []

# print(dir_list)
count = 0

for file_list in dir_list:
    print(file_list)
    # For webcam input:
    image = cv2.imread("C:/Users/sathish sp/Desktop/py.programs/Welding/Stacking/"+file_list)
    df = pd.read_csv(os.getcwd()+"/datasheet2.csv")
  
    # updating the column value/data
    df.loc[count, 'img_name'] = file_list

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
      preState = ""
      curState = ""
      #while cap.isOpened():
      '''success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")'''
        # If loading a video, use 'break' instead of 'continue'.
        #continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_height,image_width,_ = image.shape
      results = pose.process(image)
      if type(results.pose_landmarks) != type(None):
        pass
        # exit()     
        """for i in header:
          if i!="img_name":
            dict1={}
            dict1['x'] = exec("results.pose_landmarks.landmark[mp_pose.PoseLandmark."+i+"].x")
            dict1['y'] = exec("results.pose_landmarks.landmark[mp_pose.PoseLandmark."+i+"].y")
            dict1['z'] = exec("results.pose_landmarks.landmark[mp_pose.PoseLandmark."+i+"].z")
            dict1['visibility'] = exec("results.pose_landmarks.landmark[mp_pose.PoseLandmark."+i+"].visibility")
            """
        #df.loc[count, i] = str(dict1)
      '''newdf = df.copy()
      df.update(newdf)'''

      print(df)
        
      right_wrist_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width
      right_wrist_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height
      right_wrist_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z 
      right_wrist_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility
      if right_wrist_x == None: 
        right_wrist_x = '-'
      elif right_wrist_y == None:
        right_wrist_y = '-'
      elif right_wrist_z == None:
        right_wrist_z = '-'
      elif right_wrist_visibility == None:
        right_wrist_visibility = '-'
      else:
        pass          
        
      df.loc[count, 'RIGHT_WRIST'] = json.dumps({'x':right_wrist_x,'y':right_wrist_y,'z':right_wrist_z,'v':right_wrist_visibility})

      left_wrist_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width
      left_wrist_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_height
      left_wrist_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z
      left_wrist_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility
      if left_wrist_x == None: 
        left_wrist_x = '-'
      elif left_wrist_y == None:
        left_wrist_y = '-'
      elif left_wrist_z == None:
        left_wrist_z = '-'
      elif left_wrist_visibility == None:
        left_wrist_visibility = '-'
      else:
        pass 
      df.loc[count, 'LEFT_WRIST'] = json.dumps({'x':left_wrist_x,'y':left_wrist_y,'z':left_wrist_z,'v':left_wrist_visibility})


      nose_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width
      nose_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height
      nose_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z
      nose_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility
      if nose_x == None: 
        nose_x = '-'
      elif nose_y == None:
        nose_y = '-'
      elif nose_z == None:
        nose_z = '-'
      elif nose_visibility == None:
        nose_visibility = '-'
      else:
        pass
      df.loc[count, 'NOSE'] = json.dumps({'x':nose_x,'y':nose_y,'z':nose_z,'v':nose_visibility})      

      left_eye_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image_width
      left_eye_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image_height
      left_eye_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].z
      left_eye_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].visibility
      if left_eye_x == None: 
        left_eye_x = '-'
      elif left_eye_y == None:
        left_eye_y = '-'
      elif left_eye_z == None:
        left_eye_z = '-'
      elif left_eye_visibility == None:
        left_eye_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_EYE'] = json.dumps({'x':left_eye_x,'y':left_eye_y,'z':left_eye_z,'v':left_eye_visibility})

      right_eye_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image_width
      right_eye_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image_height
      right_eye_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].z 
      right_eye_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].visibility
      if right_eye_x == None: 
        right_eye_x = '-'
      elif right_eye_y == None:
        right_eye_y = '-'
      elif right_eye_z == None:
        right_eye_z = '-'
      elif right_eye_visibility == None:
        right_eye_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_EYE'] = json.dumps({'x':right_eye_x,'y':right_eye_y,'z':right_eye_z,'v':right_eye_visibility})

      left_ear_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image_width
      left_ear_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image_height
      left_ear_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z
      left_ear_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].visibility
      if left_ear_x == None: 
        left_ear_x = '-'
      elif left_ear_y == None:
        left_ear_y = '-'
      elif left_ear_z == None:
        left_ear_z = '-'
      elif left_ear_visibility == None:
        left_ear_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_EAR'] = json.dumps({'x':left_ear_x,'y':left_ear_y,'z':left_ear_z,'v':left_ear_visibility})

      right_ear_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image_width
      right_ear_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image_height
      right_ear_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z 
      right_ear_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].visibility
      if right_ear_x == None: 
        right_ear_x = '-'
      elif right_ear_y == None:
        right_ear_y = '-'
      elif right_ear_z == None:
        right_ear_z = '-'
      elif right_ear_visibility == None:
        right_ear_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_EAR'] = json.dumps({'x':right_ear_x,'y':right_ear_y,'z':right_ear_z,'v':right_ear_visibility})

      left_shoulder_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
      left_shoulder_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height
      left_shoulder_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
      left_shoulder_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
      if left_shoulder_x == None: 
        left_shoulder_x = '-'
      elif left_shoulder_y == None:
        left_shoulder_y = '-'
      elif left_shoulder_z == None:
        left_shoulder_z = '-'
      elif left_shoulder_visibility == None:
        left_shoulder_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_SHOULDER'] = json.dumps({'x':left_shoulder_x,'y':left_shoulder_y,'z':left_shoulder_z,'v':left_shoulder_visibility})


      right_shoulder_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
      right_shoulder_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height
      right_shoulder_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
      right_shoulder_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility
      if right_shoulder_x == None: 
        right_shoulder_x = '-'
      elif right_shoulder_y == None:
        right_shoulder_y = '-'
      elif right_shoulder_z == None:
        right_shoulder_z = '-'
      elif right_shoulder_visibility == None:
        right_shoulder_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_SHOULDER'] = json.dumps({'x':right_shoulder_x,'y':right_shoulder_y,'z':right_shoulder_z,'v':right_shoulder_visibility})

      left_knee_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width
      left_knee_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height
      left_knee_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z
      left_knee_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility
      if left_knee_x == None: 
        left_knee_x = '-'
      elif left_knee_y == None:
        left_knee_y = '-'
      elif left_knee_z == None:
        left_knee_z = '-'
      elif left_knee_visibility == None:
        left_knee_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_KNEE'] = json.dumps({'x':left_knee_x,'y':left_knee_y,'z':left_knee_z,'v':left_knee_visibility})

      right_knee_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width
      right_knee_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height
      right_knee_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z
      right_knee_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility
      if right_knee_x == None: 
        right_knee_x = '-'
      elif right_knee_y == None:
        right_knee_y = '-'
      elif right_knee_z == None:
        right_knee_z = '-'
      elif right_knee_visibility == None:
        right_knee_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_KNEE'] = json.dumps({'x':right_knee_x,'y':right_knee_y,'z':right_knee_z,'v':right_knee_visibility})

      left_elbow_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width
      left_elbow_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_height
      left_elbow_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z
      left_elbow_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility
      if left_elbow_x == None: 
        left_elbow_x = '-'
      elif left_elbow_y == None:
        left_elbow_y = '-'
      elif left_elbow_z == None:
        left_elbow_z = '-'
      elif left_elbow_visibility == None:
        left_elbow_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_ELBOW'] = json.dumps({'x':left_elbow_x,'y':left_elbow_y,'z':left_elbow_z,'v':left_elbow_visibility})

      right_elbow_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width
      right_elbow_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_height
      right_elbow_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z
      right_elbow_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility
      if right_elbow_x == None: 
        right_elbow_x = '-'
      elif right_elbow_y == None:
        right_elbow_y = '-'
      elif right_elbow_z == None:
        right_elbow_z = '-'
      elif right_elbow_visibility == None:
        right_elbow_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_ELBOW'] = json.dumps({'x':right_elbow_x,'y':right_elbow_y,'z':right_elbow_z,'v':right_elbow_visibility})

      right_hip_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
      right_hip_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height
      right_hip_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z
      right_hip_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility
      if right_hip_x == None: 
        right_hip_x = '-'
      elif right_hip_y == None:
        right_hip_y = '-'
      elif right_hip_z == None:
        right_hip_z = '-'
      elif right_hip_visibility == None:
        right_hip_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_HIP'] = json.dumps({'x':right_hip_x,'y':right_hip_y,'z':right_hip_z,'v':right_hip_visibility})

      left_hip_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
      left_hip_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height
      left_hip_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z
      left_hip_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility
      if left_hip_x == None: 
        left_hip_x = '-'
      elif left_hip_y == None:
        left_hip_y = '-'
      elif left_hip_z == None:
        left_hip_z = '-'
      elif left_hip_visibility == None:
        left_hip_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_HIP'] = json.dumps({'x':left_hip_x,'y':left_hip_y,'z':left_hip_z,'v':left_hip_visibility})

      right_ankle_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width
      right_ankle_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_height
      right_ankle_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z
      right_ankle_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
      if right_ankle_x == None: 
        right_ankle_x = '-'
      elif right_ankle_y == None:
        right_ankle_y = '-'
      elif right_ankle_z == None:
        right_ankle_z = '-'
      elif right_ankle_visibility == None:
        right_ankle_visibility = '-'
      else:
        pass
      df.loc[count, 'RIGHT_ANKLE'] = json.dumps({'x':right_ankle_x,'y':right_ankle_y,'z':right_ankle_z,'v':right_ankle_visibility})

      left_ankle_x= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width
      left_ankle_y= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_height
      left_ankle_z= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z
      left_ankle_visibility= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility
      if left_ankle_x == None: 
        left_ankle_x = '-'
      elif left_ankle_y == None:
        left_ankle_y = '-'
      elif left_ankle_z == None:
        left_ankle_z = '-'
      elif left_ankle_visibility == None:
        left_ankle_visibility = '-'
      else:
        pass
      df.loc[count, 'LEFT_ANKLE'] = json.dumps({'x':left_ankle_x,'y':left_ankle_y,'z':left_ankle_z,'v':left_ankle_visibility})

      df.to_csv(os.getcwd()+"\datasheet2.csv", index=False)

      '''print("---------------------------------")
      var = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
      print(type(var))
      print("-----------------------------------------------------")'''

      #df.to_csv(os.getcwd()+"/datasheet.csv", index=False)

      ''' if ( rs_y < 100):
        # print("cond 1 passes")
        if( rk_y > 200):
          curState = "person is standing"
          # print("person is standing")
      else: 
        curState = "person is sitting"
          # print("person is sitting")
      if curState != preState:
        print(curState)
  
      preState = curState'''



          
      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      
      # cv2.imwrite('/home/sathish/Desktop/skeleton/skeleton-img-'+str(i)+'.png',image)
      # Flip the image horizontally for a selfie-view display.
      # cv2.imshow('MediaPipe Pose',image)
      # i += 1
      #cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        pass
    count += 1


      #break
#cap.release()
