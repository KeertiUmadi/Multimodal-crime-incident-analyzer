import pandas as pd
 
audio  = pd.read_csv('audio/output_audio.csv')
pdf    = pd.read_csv('pdf/output_pdf.csv')
images = pd.read_csv('images/output_images.csv')
video  = pd.read_csv('video/output_video.csv')
text   = pd.read_csv('text/output_text.csv')
 
print('Audio: ',  len(audio))
print('PDF:   ',  len(pdf))
print('Images:',  len(images))
print('Video: ',  len(video))
print('Text:  ',  len(text))
print('Max:   ',  max(len(audio), len(pdf), len(images), len(video), len(text)))