from itertools import chain
from typing    import Sequence, Tuple, NamedTuple

import torch
import torch.distributions

from combat.representations import Reward

TargetState = Tuple[int, int, int] #x, y, milliseconds
GameAction  = Tuple[int, int]      #dx, dy

class GameState(NamedTuple):
    x      : int
    y      : int
    dx     : int
    dy     : int
    ddx    : int
    ddy    : int
    dddx   : int
    dddy   : int
    width  : int
    height : int
    radius : int
    targets: Tuple[int,...] #multiples of 3 consisting of target's x, y and age

class RewardFeatures:

    def to_features(self, states: Sequence[GameState]) -> torch.Tensor:

        cursors = torch.tensor([ state[0:6]                 for state in states]).float()
        boards  = torch.tensor([ [state.width,state.height] for state in states]).float()
        touched = torch.tensor(list(self._any_touched_targets(states))).float()

        l_features = cursors[:,[0,1]] / boards[:,[0,1]]
        v_features = cursors[:,[2,3]].norm(dim=1,keepdim=True) / (212.132)
        a_features = cursors[:,[4,5]].norm(dim=1,keepdim=True) / (424.264)

        d_features = cursors[:,[2,3]].float() / (v_features*212.132)
        d_features[torch.isnan(d_features)] = 0

        return touched.unsqueeze(1) * torch.hstack([l_features, v_features, a_features, d_features])

    def _any_touched_targets(self, states):

        cursors = torch.tensor([ state[0:6]     for state in states])
        radii   = torch.tensor([ [state.radius] for state in states])

        #if a state has zero targets we place a target which is impossible to touch so that all
        #calculations can then be performed naturally without any special processing logic
        states_targets = [ state.targets if state.targets else (-500,-500,50) for state in states]

        state_tps = [ list(zip(targets[0::3], targets[1::3])) for targets in states_targets ]
        state_tas = [ targets[2::3]                           for targets in states_targets ]
        state_nts = [ len(tps) for tps in state_tps ]

        cps = cursors[:,[0,1]]
        pps = cursors[:,[0,1]] - cursors[:,[2,3]]
        trs = radii

        tas = torch.tensor(list(chain.from_iterable(state_tas))).unsqueeze(1)
        tps = torch.tensor(list(chain.from_iterable(state_tps)))
        trs = trs.repeat_interleave(torch.tensor(state_nts),dim=0)
        cps = cps.repeat_interleave(torch.tensor(state_nts),dim=0)
        pps = pps.repeat_interleave(torch.tensor(state_nts),dim=0)

        cps_d_tps = cps-tps
        pps_d_tps = pps-tps

        cds = cps_d_tps.float().norm(dim=1, keepdim=True)
        pds = pps_d_tps.float().norm(dim=1, keepdim=True)

        nt = tas <= 30
        ct = cds <= trs
        pt = pds <= trs

        touched_targets = ct&(~pt|nt)

        current = 0
        for state_nt in state_nts:
            yield touched_targets[current:(current+state_nt)].any()
            current += state_nt

class ValueFeatures:

    def to_features(self, states: Sequence[GameState]) -> torch.Tensor:
        
        if isinstance(states[0], GameState):
            actions = []
        else:
            states, actions = zip(*states)

        cursors = torch.tensor([ state[0:6]                 for state in states]).float()
        boards  = torch.tensor([ [state.width,state.height] for state in states]).float()

        l_features = cursors[:,[0,1]] / boards[:,[0,1]]
        v_features = cursors[:,[2,3]].norm(dim=1, keepdim=True) / (212.132)
        a_features = cursors[:,[4,5]].norm(dim=1, keepdim=True) / (424.264)

        d_features = cursors[:,[2,3]].float() / (v_features*212)
        d_features[torch.isnan(d_features)] = 0

        t_features = self._t_features(states)

        if actions:
            return torch.hstack([l_features, v_features, a_features, d_features, t_features, torch.vstack(actions)])
        else:
            return torch.hstack([l_features, v_features, a_features, d_features, t_features])

    def _t_features(self, states):

        targets = [ state.targets if state.targets else (-500,-500,50) for state in states]
        cursors = torch.tensor([ state[0:6]     for state in states])
        radii   = torch.tensor([ [state.radius] for state in states])

        tvs = [ list(zip(t[0::3], t[1::3], t[2::3])) for t in targets ]
        tns = [ len(tv) for tv in tvs                                 ]

        cps = cursors[:,[0,1]]
        pps = cursors[:,[0,1]] - cursors[:,[2,3]]
        trs = radii

        tvs  = torch.tensor(list(chain.from_iterable(tvs)))
        tns  = torch.tensor(tns)
        
        tas = tvs[:,[2  ]]
        tps = tvs[:,[0,1]]
        
        #about a second per iteration right here
        trs = trs.repeat_interleave(tns,dim=0)
        cps = cps.repeat_interleave(tns,dim=0)
        pps = pps.repeat_interleave(tns,dim=0)

        cps_d_tps = cps-tps
        pps_d_tps = pps-tps

        cds = cps_d_tps.float().norm(dim=1, keepdim=True)
        pds = pps_d_tps.float().norm(dim=1, keepdim=True)

        nt = tas <= 30
        ct = cds <= trs
        pt = pds <= trs

        nearing_target  = (cds < pds).float()
        entering_target = (ct&(~pt|nt)).float()
        leaving_target  = (~ct&pt).float()

        indexes = torch.arange(len(states)).repeat_interleave(tns,dim=0)
        source = torch.hstack([nearing_target, entering_target, leaving_target])

        return torch.zeros(len(states),3).index_add_(0, indexes, source)

class RandomReward(Reward):

    def __init__(self):
        self._weights = torch.randn(6)
        self._feats   = RewardFeatures()

    def observe(self, state=None, states = None):
        if state is not None:
            return (self._feats.to_features([state]) @ self._weights).item()
        else:
            return (self._feats.to_features(states) @ self._weights).tolist()

