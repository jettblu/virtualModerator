import tkinter as tk
from tkinter import scrolledtext


class ModeratorWindow:
    instance = None

    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Virtual Moderator')
        self.window.geometry('1000x600')
        self.window.resizable(0, 0)

        self.top_frame = tk.Frame(self.window)
        self.top_frame.pack()

        self.label1 = tk.Label(self.top_frame, text='Moderator output:')
        self.label1.pack(side=tk.LEFT)

        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack()

        self.output_txt = scrolledtext.ScrolledText(self.main_frame)
        self.output_txt.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ModeratorWindow.instance = self

    def task():
        ModeratorWindow()
        ModeratorWindow.instance.window.mainloop()

    def appendText(text):
        if ModeratorWindow.instance is None: return
        ModeratorWindow.instance.output_txt.insert(tk.END, text + '\n')
        ModeratorWindow.instance.output_txt.see(tk.END)
