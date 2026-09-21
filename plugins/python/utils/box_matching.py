def intersection_over_union(first, second):
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["w"], second["x"] + second["w"])
    bottom = min(first["y"] + first["h"], second["y"] + second["h"])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = first["w"] * first["h"] + second["w"] * second["h"] - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def matched_box_fraction(reference_boxes_by_time, candidate_boxes_by_time, threshold):
    matched = 0
    total = 0
    for time_in_seconds, reference_boxes in reference_boxes_by_time.items():
        candidates = list(candidate_boxes_by_time.get(time_in_seconds, []))
        for reference in reference_boxes:
            total += 1
            if not candidates:
                continue
            best = max(
                range(len(candidates)),
                key=lambda index: intersection_over_union(reference, candidates[index]),
            )
            if intersection_over_union(reference, candidates[best]) >= threshold:
                matched += 1
                candidates.pop(best)
    if total == 0:
        return None
    return matched / total
