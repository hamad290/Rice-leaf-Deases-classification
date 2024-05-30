from PIL import Image
import glob

files = glob.glob('C://Users/Hamad ur rehman/Downloads/LabelledRice/Labelled/*/*')
files_reshape = list(map(lambda x: x.replace('/Labelled\\', '/Resized\\'), files))
basewidth = 300
for file, file_save in zip(files, files_reshape):
    img = Image.open(file)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize))
    img.save(file_save)