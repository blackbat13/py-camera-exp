import cv2

cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []

while True:
    ret, frame = cam.read()

    frames.append(frame.copy())

    while len(frames) > frame_height:
        frames.pop(0)

    for i in range(len(frames)):
        frame[i] = frames[i][i]

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()