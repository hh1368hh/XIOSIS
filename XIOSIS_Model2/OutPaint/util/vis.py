
import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

# Fixing random state for reproducibility
np.random.seed(19680801)


class Imshow3D:
    def __init__(self, vol, figsize=(10, 10)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # ax.invert_yaxis()
        self.fig = fig
        self.connect()
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.vol = vol
        self.vol_bk = vol

        self.maxvol = vol.max()
        self.minvol = vol.min()

        self.slices, rows, cols = vol.shape
        self.ind = self.slices // 2

        self.maxlevel = 450
        self.minlevel = -350
        self.im = ax.imshow(self.vol[self.ind, :, :], cmap='gray')
        plt.gca().invert_yaxis()
        self.sliders = widgets.IntSlider(description='Slice Number', value=self.ind, min=0, max=self.slices, step=1, )
        self.sliderl = widgets.IntSlider(description='Level', value=50, min=self.minvol, max=self.maxvol, step=1, )
        self.sliderw = widgets.IntSlider(description='Window', value=400, min=1, max=self.maxvol // 2, step=1, )
        # self.sliders =self.ind
        # l = widgets
        # ((sliders1, 'value')
        self.sliders.observe(self.handle_slider_change, names='value')

        self.sliderl.observe(self.handle_level_window, names='value')
        self.sliderw.observe(self.handle_level_window, names='value')

        display(self.sliders, self.sliderl, self.sliderw)

        self.update()

    def connect(self):

        """Connect to all the events we need."""
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)

        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.onscroll = self.fig.canvas.mpl_connect('motion_notify_event', self.on_scroll)

        self.onkey = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        # if event.button == 'up' :
        #     self.ind = (self.ind + 1) % self.slices
        # elif event.button == 'down' :
        #     self.ind = (self.ind - 1) % self.slices

        # self.update()
        True

        # if event.motion == 'up':
        #     self.ind = (self.ind + 1) % self.slices

    def on_press(self, event):
        True

    def on_release(self, event):
        True

    def on_motion(self, event):
        True

    def on_key(self, event):

        if event.key == 'up':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'down':
            self.ind = (self.ind - 1) % self.slices
        # self.fig.canvas.draw()
        self.update()

    # def previous_slice(self,ax):
    #     """Go to the previous slice."""
    #     volume = ax.volume
    #     ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    #     self.ax.images[0].set_array(volume[ax.index])

    # def next_slice(self,ax):
    #     """Go to the next slice."""
    #     volume = ax.volume
    #     ax.index = (ax.index + 1) % volume.shape[0]
    #     self.ax.images[0].set_array(volume[ax.index])

    def handle_slider_change(self, change):
        True
        # print(self.sliders.value)
        self.ind = self.sliders.value
        self.update()

    def handle_level_window(self, change):
        """
        Function to display an image slice
        Input is a numpy 2D array
        """
        self.vol = self.vol_bk

        level = self.sliderl.value
        window = self.sliderw.value
        self.maxlevel = level + window / 2
        self.minlevel = level - window / 2

        if self.maxvol < self.maxlevel:
            self.maxlevel = self.maxvol
        if self.minvol > self.minlevel:
            self.minlevel = self.minvol

        self.update()

    def update(self):
        self.sliders.value = self.ind
        self.im.set_data(self.vol[self.ind, :, :])
        self.im.set_clim(vmin=self.minlevel, vmax=self.maxlevel)
        self.ax.set_ylabel('slice %s' % self.ind)
        plt.autoscale()
        self.im.axes.figure.canvas.draw()
        # print(self.sliders.value)

# example use
# %matplotlib widget
# fig, ax = plt.subplots(1, 1, figsize=(15, 15))
#
# X = vol_arr.transpose(1, 2, 0)
# fig3D = Imshow3D(fig, ax, X)
#
# # im = ax.imshow(X[:, :, 50], cmap='gray')
# # im.cmap
#
#
# # plt.show()