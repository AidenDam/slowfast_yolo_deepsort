import cv2
from datetime import date, datetime
 
capture = cv2.VideoCapture(0)

out_file = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + '.avi'
exit_time = datetime.strptime(str(date.today()) + ' 21:30:00', r'%Y-%m-%d %H:%M:%S')

image_size = (int(capture.get(3)), int(capture.get(4)))
videoWriter = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'XVID'), 30.0, image_size)

cnt = 0
while (True): 
    try:
        ret, frame = capture.read()
     
        if ret:
	    # cv2.imshow('video', frame)
            videoWriter.write(cv2.resize(frame, image_size))
	
            cnt += 1
            if cnt >= 300:
                cnt = 0
                print("saved")
 
        if datetime.now() > exit_time:
            break
    except:
        print("\nExited!!!")
        break
 
capture.release()
videoWriter.release()
 
cv2.destroyAllWindows()

print("The video was successfully saved!") 