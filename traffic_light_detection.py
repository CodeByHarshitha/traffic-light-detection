import cv2
import os
import numpy as np

input_type = input("Enter input type (video/images/webcam): ").strip().lower()
# Folder paths for input and output type
video_path = "videos/test_dataset_video.mp4"
output_video_path = "outputs.mp4"
video_screenshots_folder = "screenshots"

image_folder = "image_test"
image_screenshots_folder = "image_screenshots"

fps = 5

#to set the min number of pixels for the object
min_area = 50
# roi
top_roi =0.1 #to ignore the top 10%
bottom_roi = 0.7 #to ignore the bottom 30%


def detect_red(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#to convert from BGR to HSV

    h,s,v = cv2.split(hsv)
    #normalizing to ensure the light detection works even in shadows or bright sunlight
    v = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)
    hsv_normalized = cv2.merge([h,s,v])

    #to calculate the avg brightness(v) and avg saturation(s) of the frame
    v_mean, s_mean = np.mean(v), np.mean(s)
    #to set min brightness and saturation threshold for red detection
    #the min brightness/saturation is set to 50 to avoid too dark pixels
    v_lower, s_lower = max(50,int(v_mean*0.6)), max(50,int(s_mean*0.5))

    #HSV color ranges for red
    lower_red1, upper_red1 = np.array([0,s_lower,v_lower]), np.array([10,255,255])
    lower_red2, upper_red2 = np.array([170,s_lower,v_lower]), np.array([179,255,255])
    #masking to turn red pixels to white and everything else black
    mask = cv2.inRange(hsv_normalized, lower_red1, upper_red1) | cv2.inRange(hsv_normalized, lower_red2, upper_red2)
    #to clean the mask and remove noise

    return mask

def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(hsv)
    v = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)
    hsv_norm = cv2.merge([h,s,v])

    v_mean, s_mean = np.mean(v), np.mean(s)
    v_lower, s_lower = max(50,int(v_mean*0.6)), max(50,int(s_mean*0.5))

    lower_green, upper_green = np.array([36,s_lower,v_lower]), np.array([85,255,255])
    mask = cv2.inRange(hsv_norm, lower_green, upper_green)

    return mask

def detect_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(hsv)
    v = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)
    hsv_norm = cv2.merge([h,s,v])

    v_mean, s_mean = np.mean(v), np.mean(s)
    v_lower, s_lower = max(50,int(v_mean*0.6)), max(50,int(s_mean*0.5))

    lower_yellow, upper_yellow = np.array([20,s_lower,v_lower]), np.array([35,255,255])
    mask = cv2.inRange(hsv_norm, lower_yellow, upper_yellow)

    return mask

def classify_light(red_mask, yellow_mask, green_mask):
    def max_area(mask):
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_a = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area and area > max_a:
                max_a = area
        return max_a

    #to calculate the largest detected area
    red_a = max_area(red_mask)
    yellow_a = max_area(yellow_mask)
    green_a = max_area(green_mask)

    if red_a > yellow_a and red_a > green_a:
        return "RED"
    elif yellow_a > red_a and yellow_a > green_a:
        return "YELLOW"
    elif green_a > red_a and green_a > yellow_a:
        return "GREEN"
    else:
        return "UNKNOWN"

def detect_lights(frame):
    #to extract just the height and width (excluding channels)
    h, w = frame.shape[:2]
    # horizontal ROI
    x1 = int(w*0.2)
    x2 = int(w*0.8)

    #vertical ROI
    y1 = int(h*top_roi)
    y2 = int(h*bottom_roi)

    #copy of the original frame
    overlay = frame.copy()

    #fill the entire overlay with dark gray color and make it look shaded
    overlay[:] = (50,50,50)
    overlay[y1:y2,x1:x2] = frame[y1:y2,x1:x2]

    #everything outside ROI is dimmed
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    #cropped ROI(only this part of the image will be analyzed)
    roi = frame[y1:y2,x1:x2]
    masks = {"RED": detect_red(roi), "YELLOW": detect_yellow(roi), "GREEN": detect_green(roi)}
    colors_bgr = {"RED": (0,0,255), "YELLOW": (0,255,255), "GREEN": (0,255,0)}

    for color, mask in masks.items():
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Loop through each detected shape in the mask
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            rect_x,rect_y,rect_width,rect_height = cv2.boundingRect(cnt)
            rect_x += x1
            rect_y += y1
            ratio = rect_width/rect_height

            if ratio <0.4 or ratio>2.0:
                continue

            cv2.rectangle(frame,(rect_x,rect_y),(rect_x+rect_width,rect_y+rect_height),colors_bgr[color],2)
            cv2.putText(frame,color,(rect_x,rect_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,colors_bgr[color],2)

    signal = classify_light(masks["RED"], masks["YELLOW"], masks["GREEN"])

    cv2.putText(frame,f"Traffic Light Signal: {signal}",(30,50),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
    return frame,signal

#For video input
if input_type=="video":
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Video could not be Opened")

    height,width = frame.shape[:2]

    #to save the output video
    out = cv2.VideoWriter(output_video_path,fourcc,fps,(width,height))


    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, state = detect_lights(frame)
        out.write(frame)
        cv2.imwrite(os.path.join(video_screenshots_folder,f"frame_{frame_number}.png"), frame)
        cv2.imshow("Traffic Light Detection", frame)
        wait_time = int(1000/fps)

        if cv2.waitKey(wait_time)&0xFF==ord('q'):
            break
        frame_number +=1

    cap.release()
    out.release()


#For image input
elif input_type=="images":
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".jpg",".png"))])

    for index,img_file in enumerate(images):
        frame = cv2.imread(os.path.join(image_folder,img_file))
        frame,state = detect_lights(frame)

        cv2.imwrite(os.path.join(image_screenshots_folder,f"annotated_{index}.png"),frame)
        cv2.imshow("Traffic Light Detection", frame)

        if cv2.waitKey(500)&0xFF==ord('q'):
            break


#input through webcam
elif input_type=="webcam":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame,state = detect_lights(frame)
        cv2.imshow("Traffic Light Detection", frame)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()
print("Processing finished.")
