import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from vcam import vcam,meshGen

WIDTH, HEIGHT = 800, 600

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

current_effect = 'dots'
def_radius = 5
strength = 0.0008

def pepare_distortion1():
    H,W = HEIGHT, WIDTH

    # Creating the virtual camera object
    c1 = vcam(H=H,W=W)

    # Creating the surface object
    plane = meshGen(H,W)

    # Defining the plane by Z = F(X,Y) 
    plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

    # Extracting the generated 3D plane
    pts3d = plane.getPlane()

    # Projecting (Capturing) the plane in the virtual camera
    pts2d = c1.project(pts3d)

    # Deriving mapping functions for mesh based warping.
    map_x,map_y = c1.getMaps(pts2d)

    return map_x, map_y

def pepare_distortion2():
    H,W = HEIGHT, WIDTH

    c1 = vcam(H=H,W=W)
    plane = meshGen(H,W)
    plane.Z += 20*np.exp(-0.5*((plane.Y*1.0/plane.H)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x,map_y = c1.getMaps(pts2d)

    return map_x, map_y

def pepare_distortion3():
    H,W = HEIGHT, WIDTH

    c1 = vcam(H=H,W=W)
    plane = meshGen(H,W)
    plane.Z -= 10*np.exp(-0.5*((plane.Y*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
    pts3d = plane.getPlane()
    pts2d = c1.project(pts3d)
    map_x,map_y = c1.getMaps(pts2d)

    return map_x, map_y

def pepare_fish_eye():
    map_x = np.zeros((HEIGHT, WIDTH), np.float32)
    map_y = np.zeros((HEIGHT, WIDTH), np.float32)

    center_x = WIDTH / 2
    center_y = HEIGHT / 2

    for y in range(HEIGHT):
        for x in range(WIDTH):
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx * dx + dy * dy)

            r = 1 + strength * (distance ** 2)
            map_x[y, x] = center_x + dx / r
            map_y[y, x] = center_y + dy / r

    return map_x, map_y

def prepare_waves():
    map_x = np.zeros((HEIGHT, WIDTH), np.float32)
    map_y = np.zeros((HEIGHT, WIDTH), np.float32)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            map_x[y,x] = x + 30*np.sin(y/30)
            map_y[y,x] = y + 30*np.cos(x/30)
    return map_x, map_y

distortion1_map_x, distortion1_map_y = pepare_distortion1()
distortion2_map_x, distortion2_map_y = pepare_distortion2()
distortion3_map_x, distortion3_map_y = pepare_distortion3()
fish_eye_map_x, fish_eye_map_y = pepare_fish_eye()
waves_map_x, waves_map_y = prepare_waves()

def dots_effect(image):
    cell_width, cell_height = 12, 12
    new_width, new_height = int(WIDTH / cell_width), int(HEIGHT / cell_height)

    black_window = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    small_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    for i in range(new_height):
        for j in range(new_width):
            color = small_image[i, j]
            coord = (j * cell_width + cell_width, i * cell_height)
            cv2.circle(black_window, coord, def_radius, tuple(int(c) for c in color), 2)

    return black_window

def fish_eye_effect(image):
    return cv2.remap(image, fish_eye_map_x, fish_eye_map_y, interpolation=cv2.INTER_LINEAR)

    
def distortion1_effect(image):
    return cv2.remap(image, distortion1_map_x, distortion1_map_y, interpolation=cv2.INTER_LINEAR)

def distortion2_effect(image):
    return cv2.remap(image, distortion2_map_x, distortion2_map_y, interpolation=cv2.INTER_LINEAR)

def distortion3_effect(image):
    return cv2.remap(image, distortion3_map_x, distortion3_map_y, interpolation=cv2.INTER_LINEAR)

def mirror_maze_effect(image):
    return cv2.hconcat([cv2.flip(image,1), image])

def waves_effect(image):
    return cv2.remap(image, waves_map_x, waves_map_y, interpolation=cv2.INTER_LINEAR)

def pixelate_effect(image):
    h, w = image.shape[:2]
    small = cv2.resize(image, (w//strength, h//strength))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def outline_effect(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def retro_game_effect(image, color_depth=4):
    quantized = image // (256//color_depth) * (256//color_depth)
    return cv2.resize(cv2.resize(quantized, (64,48)), image.shape[1::-1], interpolation=cv2.INTER_NEAREST)


def toggle_slider_visibility(effect):
    if effect in ['dots', 'fish_eye', 'pixelate']:
        video_label.pack_forget()
        slider.pack(fill=tk.X, padx=10, pady=10)
        video_label.pack()
        # slider.lift()
    else:
        slider.pack_forget()

def change_effect(new_effect):
    global current_effect
    current_effect = new_effect
    toggle_slider_visibility(new_effect)
    if new_effect == 'dots':
        slider.set(def_radius)
    elif new_effect == 'fish_eye':
        slider.set(50)
    elif new_effect == "pixelate":
        slider.set(10)
    else:
        slider.set(0)

def update_slider(val):
    global def_radius, strength, fish_eye_map_x, fish_eye_map_y 
    val = int(float(val))
    if current_effect == 'dots':
        def_radius = max(1, val)
    elif current_effect == 'fish_eye':
        strength = (val - 90) * (0.00009 / 90)
        fish_eye_map_x, fish_eye_map_y = pepare_fish_eye()
    elif current_effect == 'pixelate':
        strength = max(1, val)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    if current_effect == 'dots':
        out = dots_effect(frame)
    elif current_effect == 'fish_eye':
        out = fish_eye_effect(frame)
    elif current_effect == 'distortion1':
        out = distortion1_effect(frame)
    elif current_effect == 'distortion2':
        out = distortion2_effect(frame)
    elif current_effect == 'distortion3':
        out = distortion3_effect(frame)
    elif current_effect == 'mirror_maze':
        out = mirror_maze_effect(frame)
    elif current_effect == 'waves':
        out = waves_effect(frame)
    elif current_effect == 'pixelate':
        out = pixelate_effect(frame)
    elif current_effect == 'outline':
        out = outline_effect(frame)
    elif current_effect == 'retro_game':
        out = retro_game_effect(frame)
    else:
        out = frame

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(out)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    img = img.resize((screen_width, screen_height))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)


root = tk.Tk()
root.title("Efekty na kamerze")

button_frame = tk.Frame(root)
button_frame.pack()

btn_dots = tk.Button(button_frame, text="Kropki", width=20, command=lambda: change_effect('dots'))
btn_dots.pack(side=tk.LEFT)

btn_fish_eye = tk.Button(button_frame, text="Rybie Oko", width=20, command=lambda: change_effect('fish_eye'))
btn_fish_eye.pack(side=tk.LEFT)

btn_distortion1 = tk.Button(button_frame, text="Krzywe zwierciadło", width=20, command=lambda: change_effect('distortion1'))
btn_distortion1.pack(side=tk.LEFT)

btn_distortion2 = tk.Button(button_frame, text="Duża głowa", width=20, command=lambda: change_effect('distortion2'))
btn_distortion2.pack(side=tk.LEFT)

btn_distortion3 = tk.Button(button_frame, text="Klepsydra", width=20, command=lambda: change_effect('distortion3'))
btn_distortion3.pack(side=tk.LEFT)

btn_mirror_maze = tk.Button(button_frame, text="Symetria", width=20, command=lambda: change_effect('mirror_maze'))
btn_mirror_maze.pack(side=tk.LEFT)

btn_waves = tk.Button(button_frame, text="Fale", width=20, command=lambda: change_effect('waves'))
btn_waves.pack(side=tk.LEFT)

btn_pixelate = tk.Button(button_frame, text="Piksele", width=20, command=lambda: change_effect('pixelate'))
btn_pixelate.pack(side=tk.LEFT)

btn_outline = tk.Button(button_frame, text="Obrys", width=20, command=lambda: change_effect('outline'))
btn_outline.pack(side=tk.LEFT)

btn_retro_game = tk.Button(button_frame, text="Gra retro", width=20, command=lambda: change_effect('retro_game'))
btn_retro_game.pack(side=tk.LEFT)

slider = ttk.Scale(root, from_=0, to=180, orient=tk.HORIZONTAL, command=update_slider)
slider.pack(fill=tk.X, padx=10, pady=10)
slider.set(5)

video_label = tk.Label(root)
video_label.pack()

update_frame()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()
