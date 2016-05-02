import gtk
import pamImage
import croplib
import wordio
import word


class Annotator(object):
    def __init__(self, word_file, img_file, out_file):
        self.out_file = out_file

        win = gtk.Window(gtk.WINDOW_TOPLEVEL)
        win.set_size_request(500, 250)
        win.connect('destroy', lambda w: gtk.main_quit())
        vbox = gtk.VBox()

        self.new_lines = [[]]  # store the annotated words + chars here

        self.lines, _ = wordio.read(word_file)
        self.img = pamImage.PamImage(img_file)

        # Keep track of where we are in the file
        self.line_iter = iter(self.lines)
        self.cur_line = self.line_iter.next()
        self.word_iter = iter(self.cur_line)

        self.pb = None
        self.word = None
        self.cropped = None

        # Cursor and annotation-points (x-coordinates)
        self.current_x = 0
        self.points = []

        # Drawing area to draw the lines on
        self.drawing_area = gtk.DrawingArea()
        self.drawing_area.connect("expose-event", self.on_expose)
        self.drawing_area.connect("motion_notify_event", self.on_motion)
        self.drawing_area.connect("button_press_event",
                                  self.on_mouse_button_press)
        self.drawing_area.set_events(gtk.gdk.EXPOSURE_MASK
                                     | gtk.gdk.LEAVE_NOTIFY_MASK
                                     | gtk.gdk.BUTTON_PRESS_MASK
                                     | gtk.gdk.POINTER_MOTION_MASK
                                     | gtk.gdk.POINTER_MOTION_HINT_MASK)

        vbox.pack_start(self.drawing_area)

        hbox = gtk.HBox()
        vbox.pack_start(hbox)

        # Label row
        label = gtk.Label()
        label.set_text("Characters labeled above:")
        hbox.pack_start(label)
        self.entry = gtk.Entry()
        hbox.pack_start(self.entry)

        # control button row
        self.reset_button = gtk.Button("Reset")
        self.next_button = gtk.Button("Next")
        self.reset_button.connect("clicked", self.on_reset)
        self.next_button.connect("clicked", self.on_next)
        hbox = gtk.HBox()
        hbox.pack_start(self.reset_button)
        hbox.pack_start(self.next_button)

        vbox.pack_start(hbox)

        self.next_word()

        win.add(vbox)
        win.show_all()

    def make_pixbuf(self, im):
        """Creates a pixbuf from a PamImage object for display on the screen"""
        # First store it as a large string. This might have been more efficient
        # in a C-function, but it looks like it doesn't matter too much here.
        pbdata = ""
        for row in range(im.getHeight()):
            for col in range(im.getWidth()):
                px = im.getPixelRGB(col, row)
                pbdata += chr(px.r) + chr(px.g) + chr(px.b)

        self.pb = gtk.gdk.pixbuf_new_from_data(pbdata, gtk.gdk.COLORSPACE_RGB,
                                               False, 8, im.getWidth(), im.getHeight(),
                                               im.getWidth() * 3)
        return self.pb

    def set_drawing_size(self, im):
        """Drawing area needs the same width and height as the current word"""
        self.drawing_area.set_size_request(im.getWidth(), im.getHeight());

    def next_word(self):
        """Set the next word, or stop the program if we're done."""
        try:
            self.word = self.word_iter.next()
        except StopIteration:
            try:
                self.cur_line = self.line_iter.next()
                self.new_lines.append([])
            except StopIteration:
                self.save_quit()
            self.word_iter = iter(self.cur_line)
            self.word = self.word_iter.next()
        self.cropped = croplib.crop(self.img, self.word.left, self.word.top,
                                    self.word.right, self.word.bottom)
        self.make_pixbuf(self.cropped)
        self.set_drawing_size(self.cropped)
        self.entry.set_text(self.word.text)

    def on_expose(self, widget, event):
        """This is the drawing routine.

        It is called every time the screen is redrawn, and used to draw the
        current word, the stored annotation coordinates and the current
        cursor"""

        if not self.cropped:
            return

        drawable = widget.window
        drawable.draw_pixbuf(widget.get_style().fg_gc[gtk.STATE_NORMAL],
                             self.pb, 0, 0, 0, 0)
        widget.queue_draw()
        gc = widget.get_style().fg_gc[gtk.STATE_NORMAL]

        # Draw cursor
        gc.set_rgb_fg_color(gtk.gdk.color_parse("black"))
        drawable.draw_line(gc,
                           self.current_x, 0, self.current_x, self.cropped.getHeight())

        # Draw the previous points
        gc.set_rgb_fg_color(gtk.gdk.color_parse("red"))
        for x in self.points:
            drawable.draw_line(gc, x, 0, x, self.cropped.getHeight())

        # set the foreground color back to black
        gc.set_rgb_fg_color(gtk.gdk.color_parse("black"))

    def on_motion(self, widget, event):
        """Store cursor position"""
        if event.is_hint:
            x, y, state = event.window.get_pointer()
        else:
            x = event.x
        self.current_x = x

    def on_mouse_button_press(self, widget, event):
        """Store coordinate under mouse cursor"""
        self.points.append(int(event.x))

    def on_reset(self, e):
        """Start over."""
        del self.points[:]

    def on_next(self, e):
        """The next button is pressed: store current word and move on."""
        txt = self.entry.get_text()

        if txt:
            print len(txt)
            print len(self.points)
            print self.points
            assert len(txt) == len(self.points) + 1

            self.points.sort()
            pts_ = [0] + self.points + [self.cropped.getWidth()]
            pts = zip(pts_[:-1], pts_[1:])
            for c, pt in zip(txt, pts):
                char = word.Character()
                char.shear = 0
                char.text = c
                char.top = self.word.top
                char.bottom = self.word.bottom
                char.left = self.word.left + pt[0]
                char.right = self.word.left + pt[1]
                self.word.characters.append(char)

        self.new_lines[-1].append(self.word)
        self.next_word()
        self.on_reset(e)  # don't forget to clear out the annotation points.

    def save_quit(self):
        """Save and quit."""
        wordio.save(self.new_lines, self.out_file)
        print "Done."
        gtk.main_quit()


def main():
    import sys
    if len(sys.argv) != 4:
        print "Usage: %s <image> <input .words file> <output .words file>" % sys.argv[0]
        print
        print """Click between the boundaries of characters to annotate characters.
The word in the text entry box are the characters that you annotated. Please
annotate the characters you *see*, not the intended characters. So, for
example, `@nos_nof' should now read `nof' in the entry box. Supply the right
amount of characters (i.e., |segmentation_points| + 1)!"""
        print
        print "**NOTE**: Only saves when reaching the end of the file."
        sys.exit(1)
    Annotator(sys.argv[2], sys.argv[1], sys.argv[3])
    gtk.main()


if __name__ == "__main__":
    main()
