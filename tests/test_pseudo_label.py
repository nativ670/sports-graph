from sportsgraph.detection.pseudo_label import format_yolo_line, pseudo_label_lines


def test_format_yolo_line():
    assert format_yolo_line(0, 0.5, 0.25, 0.1, 0.2) == "0 0.500000 0.250000 0.100000 0.200000"


def test_pseudo_label_lines_keeps_relevant_classes_above_threshold():
    detections = [
        ("person", (0.5, 0.5, 0.1, 0.2), 0.9),  # player, confident -> kept
        ("sports ball", (0.1, 0.1, 0.02, 0.02), 0.6),  # ball, confident -> kept
        ("car", (0.8, 0.8, 0.3, 0.1), 0.95),  # irrelevant class -> dropped
        ("person", (0.2, 0.2, 0.1, 0.1), 0.1),  # player, but below threshold -> dropped
    ]
    our_class_name_to_id = {"player": 0, "ball": 1}

    lines = pseudo_label_lines(detections, our_class_name_to_id, conf_threshold=0.25)

    assert lines == [
        "0 0.500000 0.500000 0.100000 0.200000",
        "1 0.100000 0.100000 0.020000 0.020000",
    ]


def test_pseudo_label_lines_empty_when_nothing_qualifies():
    detections = [("car", (0.5, 0.5, 0.1, 0.1), 0.99)]
    assert pseudo_label_lines(detections, {"player": 0, "ball": 1}) == []


def test_pseudo_label_lines_respects_custom_class_ordering():
    # Confirms the id mapping comes from the classes file, not a hardcoded 0/1 --
    # if someone's CVAT project added "ball" before "player", this must still work.
    detections = [("person", (0.5, 0.5, 0.1, 0.1), 0.9)]
    our_class_name_to_id = {"ball": 0, "player": 1}  # reversed order
    lines = pseudo_label_lines(detections, our_class_name_to_id, conf_threshold=0.25)
    assert lines == ["1 0.500000 0.500000 0.100000 0.100000"]
