"""
Load and save words to/from XML (.words) files
Unicode improvements by Twan van Laarhoven.
"""

import os
import xml.sax.handler
import xml.sax
from xml.sax.saxutils import escape
from word import Word, Character
import codecs


def avg(items):
    """ Return average value """
    return sum(items) / len(items)


class WordLayoutReader(xml.sax.handler.ContentHandler):
    """
    XML reader derived from Sax, a Python XML reader
    """

    def __init__(self):
        self.lines = []  # each line consists of words

    # self.words = [] # Initialize an empty list of rectangles

    def startElement(self, name, attrs):
        if name == "Image":
            self.image_name = str(attrs['name'])

        if name == "TextLine":
            self.cur_line = []

        if name == "Word":
            word = Word()
            word.top = int(attrs['top'])
            word.bottom = int(attrs['bottom'])
            word.left = int(attrs['left'])
            word.right = int(attrs['right'])
            word.text = unicode(attrs['text'])
            word.shear = int(attrs['shear'])
            self.cur_line.append(word)

        if name == "Character":
            char = Character()
            if 'top' in attrs:
                char.top = int(attrs['top'])
            else:
                char.top = self.cur_line[-1].top
            if 'bottom' in attrs:
                char.bottom = int(attrs['bottom'])
            else:
                char.bottom = self.cur_line[-1].bottom
            char.left = int(attrs['left'])
            char.right = int(attrs['right'])
            char.text = unicode(attrs['text'])
            if 'shear' in attrs:
                char.shear = int(attrs['shear'])
            else:
                char.shear = self.cur_line[-1].shear
            self.cur_line[-1].characters.append(char)

    def endElement(self, name):
        if name == "TextLine":
            self.lines.append(self.cur_line)
            self.cur_line = []

    def get_lines(self):
        return self.lines

    def get_image_name(self):
        return self.image_name


def read(xmlfile):
    """ Return lines (containing word objects) and image name from XML file """
    reader = WordLayoutReader()
    xml.sax.parse(xmlfile, reader)
    return reader.get_lines(), reader.get_image_name()


def save(word_lines, xml_file):
    """ Save lines (containing words) to XML file """
    file_id = os.path.basename(os.path.splitext(xml_file)[0])
    file = codecs.open(xml_file, 'w', 'utf-8')
    file.write('<?xml version="1.0" encoding = "UTF-8"?>\n')
    file.write('<Image name="%s">\n' % file_id)
    line_num = 0
    for word_line in word_lines:
        if len(word_line) > 0:
            line_num += 1
            top = min([word.top for word in word_line])
            bottom = max([word.bottom for word in word_line])
            left = min([word.left for word in word_line])
            right = max([word.right for word in word_line])
            shear = avg([word.shear for word in word_line])
            file.write('    <TextLine no="%i" top="%i" bottom="%i" left="%i" right="%i" shear="%i">\n' %
                       (line_num, top, bottom, left, right, shear))
            word_num = 0
            for word in word_line:
                word_num += 1
                file.write('        <Word no="%i" top="%i" bottom="%i" left="%i" right="%i" shear="%i" text="%s"' %
                           (word_num, word.top, word.bottom, word.left, word.right, word.shear, escape(word.text)))
                if not word.characters:
                    file.write('/>\n')
                else:
                    file.write('>\n')
                    for i, char in enumerate(word.characters):
                        if not char.shear:  char.shear = word.shear
                        if not char.top:    char.top = word.top
                        if not char.bottom: char.bottom = word.bottom
                        file.write(
                            '            <Character no="%i" top="%i" bottom="%i" left="%i" right="%i" shear="%i" text="%s"/>\n' %
                            (i + 1, char.top, char.bottom, char.left, char.right, char.shear, escape(char.text)))
                    file.write('        </Word>\n')

            file.write('    </TextLine>\n')
    file.write('</Image>\n')
    file.close()
    print "File %s saved." % xml_file
