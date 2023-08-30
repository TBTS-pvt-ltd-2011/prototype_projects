
import sys
from tkinter import *
import time
 
def timing():
    current_time = time.strftime("%H : %M : %S")
    clock.config(text=current_time)
    #inserting a 200 seconds difference will output the exact time
    clock.after(200,timing)
 
root=Tk()
root.geometry("100x100")
clock=Label(root,font=("times",60,"bold"),bg="grey")
clock.grid(row=3,column=3,pady=20,padx=100)
timing()
 
digital=Label(root,font="arial")
digital.grid(row=2,column=2)
 
nota=Label(root,text="",font="arial")
nota.grid(row=3,column=2)
 
root.mainloop()