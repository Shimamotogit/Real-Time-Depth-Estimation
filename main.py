from tkinter import messagebox
import tkinter.ttk as ttk
import tkinter as tk
import PIL.Image, PIL.ImageTk
import detection as dt
import cv2
# from PIL import Image, ImageTk, ImageOps

# from main import detection, detection_image
#https://torimakujoukyou.com/python-opencv-movie-tkinter/
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master.geometry("850x500+50+50")
        self.master.title("単眼カメラの相対距離推定")

        self.target_names = []
        self.variable = tk.StringVar()
        self.names_lst = ['person', 'bottle', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                          'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
                          'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                          'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                          'hair drier', 'toothbrush']

        self.detection = dt.detection()
        self.WIDTH, self.HEIGHT = self.detection.set_capture()
        self.delay = int(10)

        self.create_frame()
        self.play_video()
        self.create_widget()

        print(self.HEIGHT, self.WIDTH)

    def create_frame(self):

        self.frame1 = tk.Frame(self.master, width=600, height=350, bg="#E6E6E6")
        self.frame1.place(x=0, y=0)

        self.frame2 = tk.Frame(self.master, width=700, height=100, relief="flat")
        self.frame2.place(x=0, y=400)
        print_target_name = tk.Label(self.frame2, text= "検出対象：", relief="flat", font=20)
        print_target_name.grid(row=0, column=0, sticky = tk.W)
        self.target_label = tk.Label(self.frame2, text="", relief="flat", font=20)
        self.target_label.grid(row=1, column=0, sticky = tk.W)

        self.frame3 = tk.Frame(self.master, width=600, height=400, bg="#E6E6E6")
        self.frame3.place(x=700, y=0)

        self.canvas = tk.Canvas(self.frame1, width=700, height=450)
        self.canvas.pack()

    def print_targets(self):
        self.target_label["text"] = ", ".join(self.target_names)
        # l_property = tk.Label(self.frame2,text=", ".join(self.target_names), relief="flat", font=20)
        # l_property.grid(row=1, column=0, sticky = tk.W)

    def add_target_name(self):
        add_name = str(self.variable.get())
        if add_name == "":
            messagebox.showerror("警告", f"検出対象が選択されていません。")
            # print("検出対象を入力してください")
            return

        if not add_name in self.target_names:
            self.target_names.append(add_name)
            print(self.target_names)
            self.print_targets()

        else:
            messagebox.showerror("警告", f"{add_name}はすでに登録されています。")

    def delete_target_name(self):
        delete_name = str(self.variable.get())
        if delete_name in self.target_names:
            self.target_names.remove(delete_name)
            print(self.target_names)
            self.print_targets()

        else:
            messagebox.showerror("警告", f"{delete_name}は登録されていません")

    def create_widget(self):
        l_property = tk.Label(self.frame3,text="検出対象を設定　　", font=20, relief="flat")
        l_property.grid(row=0, column=0, sticky = tk.W)

        frame3_in_frame = tk.LabelFrame(self.frame3, text="", relief="flat")
        frame3_in_frame.grid(row=7, column=0, sticky = tk.W)
        
        self.pludown_ = ttk.Combobox(self.frame3 , values = self.names_lst , textvariable = self.variable ).grid(row=6, column=0, sticky = tk.W)
        
        tk.Button(frame3_in_frame, text="追加", command=self.add_target_name).grid(row=0, column=0, sticky = tk.W)
        tk.Button(frame3_in_frame, text="削除", command=self.delete_target_name).grid(row=0, column=1, sticky = tk.W)
        
        # frame3_in_frame = tk.LabelFrame(self.frame3, text="", relief="flat")
        # frame3_in_frame.grid(row=8, column=0, sticky = tk.W)
        # tk.Button(frame3_in_frame, text="処理開始", command=self.play_video).grid(row=0, column=0, sticky = tk.W)
        # tk.Button(frame3_in_frame, text="処理停止", command=self.play_video).grid(row=0, column=1, sticky = tk.W)
        
    def play_video(self):
        frame = self.detection.detection_image(target_names=self.target_names)
        frame = cv2.resize(frame, dsize=(700, int(700*self.HEIGHT/self.WIDTH)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        self.canvas.create_image(0,0, image= self.photo, anchor = tk.NW)

        self.master.after(self.delay, self.play_video)

def main():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()