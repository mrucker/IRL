from itertools import chain
from typing import Sequence, Tuple

import matplotlib.pyplot    as plt #type: ignore
import matplotlib.animation as ani #type: ignore

from matplotlib            import rcParams        #type: ignore
from matplotlib.artist     import Artist          #type: ignore
from matplotlib.lines      import Line2D          #type: ignore
from matplotlib.text       import Text            #type: ignore
from matplotlib.transforms import TransformedBbox #type: ignore
from matplotlib.image      import BboxImage       #type: ignore

from combat.config import OUT_PATH

class TransformableBboxImage(BboxImage):

    def set_transform(self, t):
        self.bbox = TransformedBbox(self.bbox, t)
        super().set_transform(t)

class VideoPlot:

    def __init__(self, 
        title:str,
        frames1: Sequence[Sequence[Artist]], 
        frames2: Sequence[Sequence[Artist]] = None, 
        margin = .05) -> None:

        self._title   = title
        self._rwd_res = (40,40)
        self._frames1 = frames1
        self._frames2 = frames2
        self._margin  = margin

    def _make_anmiation(self) -> Tuple[plt.Figure, Sequence[Sequence[Artist]]]:
                
        rwd_w = 1 / (self._rwd_res[0]-1)
        rwd_h = 1 / (self._rwd_res[1]-1)
        
        xlim = (0-(rwd_w/2), 1+(rwd_w/2))
        ylim = (0-(rwd_h/2), 1+(rwd_h/2))

        x_margin = (xlim[1] - xlim[0]) * self._margin
        y_margin = (ylim[1] - ylim[0]) * self._margin
        
        x_offset = 0

        if self._frames2 is not None:
            fig = plt.figure(figsize=(10,5))
        else:
            fig = plt.figure(figsize=(7,5))

        fig.suptitle(self._title)

        if self._frames2 is not None:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None

        for ax, frames, title in [ (ax1, self._frames1, 'reward/expert'), (ax2, self._frames2, 'Q/learned')]:

            if ax is None: continue

            for artist in chain.from_iterable(frames):
                if isinstance(artist, Text):
                    artist.set_transform(ax.transAxes)
                    fig.add_artist(artist)
                else:
                    artist.set_transform(ax.transData)
                    ax.add_artist(artist)

            ax.set_title(title)

            ax.set_xlim(xlim[0]-x_margin,xlim[1]+x_margin)
            ax.set_ylim(ylim[0]-y_margin,ylim[1]+y_margin)

            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            box = ax.get_position()
            ax.set_position([box.x0-x_offset, box.y0, box.width * 0.85, box.height])
            x_offset += box.width * 0.15

        legend_elements = [
            Line2D([0], [0], marker='o', label='active' , color='gray', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='x', label='casualty', color='gray', markersize=8, linestyle='None')
        ]

        if ax2 is not None:
            ax2.legend(handles=legend_elements, framealpha=1, facecolor="white", loc="upper left", bbox_to_anchor = (1.02,.99), bbox_transform=ax2.transAxes)
        else:
            ax1.legend(handles=legend_elements, framealpha=1, facecolor="white", loc="upper left", bbox_to_anchor = (1.02,.99), bbox_transform=ax1.transAxes)

        if self._frames2 is not None:
            frames = [ list(chain(f1,f2)) for f1,f2 in zip(self._frames1, self._frames2) ]
        else:
            frames = self._frames1

        return fig, frames

    def show(self, block=False) -> None:
        print(f"showing animation {self._title} ...")

        figure, frames = self._make_anmiation()

        #we stick it in a variables to make sure it isn't GC'd before showing
        animation = ani.ArtistAnimation(figure, frames, interval=250, blit=False)

        plt.show(block=block)
        plt.close(figure)

    def save(self, filename:str, lock=None) -> None:
        print(f"saving animation {self._title} to {filename}.mp4 ...")

        figure, frames = self._make_anmiation()

        #we stick it in a variables to make sure it isn't GC'd before showing
        animation = ani.ArtistAnimation(figure, frames, interval=250, blit=False)

        try:
            if lock is not None: lock.acquire()
            animation.save(f"{OUT_PATH}/{filename}.mp4")
        finally:
            if lock is not None: lock.release()

        plt.close(figure)

class StaticPlot:

    def __init__(self, 
        title:str, 
        fig1: Sequence[Artist],
        fig2: Sequence[Artist],
        margin = .05) -> None:

        self._title  = title
        self._fig1   = fig1
        self._fig2   = fig2
        self._margin = margin

    def _make_figure(self) -> plt.Figure:

        xlim = (0,1)
        ylim = (0,1)

        rcParams.update({'figure.autolayout': True})

        x_margin = (xlim[1] - xlim[0]) * self._margin
        y_margin = (ylim[1] - ylim[0]) * self._margin

        fig = plt.figure(figsize=(6,4))

        ax = fig.add_subplot(111)

        ax.set_title(self._title)

        for artist in self._fig1:
            artist.set_color('#1E88E5')
            artist.set_transform(ax.transData)
            ax.add_artist(artist)

        for artist in self._fig2:
            artist.set_color('#FFC107')
            artist.set_transform(ax.transData)
            ax.add_artist(artist)

        ax.set_xlim(xlim[0]-x_margin,xlim[1]+x_margin)
        ax.set_ylim(ylim[0]-y_margin,ylim[1]+y_margin)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        legend_elements = [
            Line2D([0], [0], label='Expert Behavior' , color='#1E88E5'),
            Line2D([0], [0], label='Learned Behavior', color='#FFC107')
        ]

        ax.legend(handles=legend_elements, framealpha=1, facecolor="white")

        return fig

    def show(self) -> None:
        print(f"showing trajectory {self._title} ...")

        figure = self._make_figure()

        plt.show()
        plt.close(figure)

    def save(self, filename:str) -> None:
        print(f"saving trajectory {self._title} to {filename}.mp4 ...")

        figure = self._make_figure()
        
        plt.savefig(f"{OUT_PATH}/{filename}.png")
        plt.close(figure)
