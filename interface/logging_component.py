import tkinter as tk
from datetime import datetime
from interface.styling import *

class Logging(tk.Frame):
    def __init__(self, *args, **kwargs):  # Fixed the missing colon here
     super().__init__(*args, **kwargs)

     self.logging_text = tk.Text(self, height=10,width=60,state=tk.DISABLED,bg=BG_COLOR, fg=FG_COLOR_2, font=GLOBAL_FONT)
     self.logging_text.pack(side=tk.TOP)

    def add_log(self,message:str):
        self.logging_text.configure(state=tk.NORMAL)
        self.logging_text.insert("1.0",datetime.now().strftime("%a %H:%M:%S ::") +message+ "\n")  #tk.END instead of 1.0 will print at the end
        self.logging_text.configure(state=tk.DISABLED)