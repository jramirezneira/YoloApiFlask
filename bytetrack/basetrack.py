import numpy as np
from collections import OrderedDict, Counter


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0
    elemnts = []
    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        # BaseTrack.elemnts.append(cls)
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the global track ID counter."""
        BaseTrack._count = 0
        # BaseTrack.elemnts = []

    # @staticmethod
    # def get_stats():
    #     return Counter(BaseTrack.elemnts)
