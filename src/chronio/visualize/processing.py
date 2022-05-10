from typing import Dict, Any, List


def build_coords_dict(event_onsets: Dict[str, list], event_durations: Dict[str, Any]) -> Dict[str, List[tuple]]:
    """
    Given event onset and event duration information, obtain a dict containing the x coordinates for each event type.

    :param event_onsets:    A dict whose keys are event names and whose keys are lists of event onset times (in seconds)
    :type event_onsets:     Dict[str, list]

    :param event_durations: A dict whose keys match that of event_onsets and whose keys are either scalars or lists. If
                            keys are scalar, each event of that type is assumed to have the same duration. Keys that
                            are lists permit events of the same type to vary in length.
    :type event_durations:  Dict[str, Any]

    :return:
    """
    coords_dict = {}
    for i, (ttype, onsets) in enumerate(event_onsets.items()):
        coords_dict[ttype] = []

        if type(event_durations[ttype]) == list:
            for onset, dur in zip(onsets, event_durations[ttype]):
                coords_dict[ttype].append((onset, onset + dur))

        else:
            dur = event_durations[ttype]
            for onset in onsets:
                coords_dict[ttype].append((onset, onset + dur))

    return coords_dict
