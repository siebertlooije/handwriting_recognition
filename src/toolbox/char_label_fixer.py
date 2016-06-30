import PIL.Image as Image
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg") # This program works with Qt only
import pylab as pl
fig, ax1 = pl.subplots()

out_file = open('../toolbox/fixed_labels.txt', 'w')


### control panel ###
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt

total = 16954 + 12

paths = []
labels = []
for i, line in enumerate(open('../toolbox/labels.txt')):
    #if i < 12:
    #    continue
    path, label = line.split()
    paths.append(path)
    labels.append(label)

root = fig.canvas.manager.window
panel = QtGui.QWidget()
hbox = QtGui.QHBoxLayout(panel)
textbox = QtGui.QLineEdit(parent=panel)

idx = 0

initial_len = len(paths)

print initial_len

def update():
    text = textbox.text()
    if text != '':
        out_file.write(paths[idx] + ' ' + textbox.text() + '\n')
    path = paths.pop()
    label = labels.pop()
    im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')))
    ax1.imshow(im)
    ax1.set_title(chr(int(label)) + ', iter' + str(initial_len - len(paths)))
    textbox.setText(chr(int(label)))
    fig.canvas.draw_idle()

path = paths.pop()
label = labels.pop()
im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')))
ax1.imshow(im)
ax1.set_title(chr(int(label)) + ', iter' + str(initial_len - len(paths)))
textbox.setText(chr(int(label)))
fig.canvas.draw_idle()


#textbox.textChanged.connect(update)
#enter = QtGui.QKeyEvent()
textbox.returnPressed.connect(update)
hbox.addWidget(textbox)
panel.setLayout(hbox)

dock = QtGui.QDockWidget("control", root)
root.addDockWidget(Qt.BottomDockWidgetArea, dock)
dock.setWidget(panel)
######################

pl.show()

out_file.close()

