from tkinter import *
from PIL import ImageTk, Image
from src.model.ACGAN.ACGAN import ACGAN
from src.model.PROGAN.PROGAN import PROGAN

window=Tk()
window.iconbitmap(r'icons/top.ico')
img = ImageTk.PhotoImage(Image.open("icons/sideimg.jpeg"))
Label(image=img).place(x=0, y=0, width=300, height=400)
lbl=Label(window, text="Text Description To Realistic Face Synthesis", fg='red', font=("Helvetica", 16))
lbl.place(x=365, y=50)
lbl2=Label(window, text="Using Progressive Growing GAN", fg='red', font=("Helvetica", 16))
lbl2.place(x=420, y=80)
lbl3=Label(window, text="Please Enter The Facial Description", fg='blue', font=("Helvetica", 12))
lbl3.place(x=385, y=130)
lbl4=Label(window, text="Output", fg='blue', font=("Helvetica", 12))
lbl4.place(x=735, y=130)
txtfld=Text(window, bd=5, width="42", height=7)
txtfld.place(x=340, y=160)
face = ImageTk.PhotoImage(Image.open("icons/black.png"))
faceLabel = Label(image=face)
faceLabel.place(x=700, y=160, width=122, height=122)
def generateImage():
    global face
    global faceLabel
    txt = txtfld.get("1.0", "end-1c")
    print("Doing.......",txt)
    flag = ACGAN.generateImage(txt)
    if(flag==1):
        PROGAN.convertToHighResolutionImage('results/ACGAN_out.png','results/PROGAN_out.png')
        face = ImageTk.PhotoImage(Image.open('results/PROGAN_out.png'))
        faceLabel = Label(image=face)
        faceLabel.place(x=700, y=160, width=122, height=122)
btn=Button(window, text="GENERATE THE FACE", fg='green', font=("Helvetica", 12, 'bold'), command=generateImage)
btn.place(x=420, y=300, width=300, height=40)
window.title('Text Description To Realistic Face Synthesis')
window.geometry("850x400+10+10")
window.mainloop()