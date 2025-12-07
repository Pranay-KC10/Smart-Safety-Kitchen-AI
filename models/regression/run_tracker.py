import math

class PositionTracker:
    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ( (x1 + x2) / 2, (y1 + y2) / 2 )

    def distance(self, p1, p2):
        return math.dist(p1, p2)

    def compute_person_stove_distance(self, person_bbox, stove_bbox):
        if person_bbox is None or stove_bbox is None:
            return None
        p = self.get_center(person_bbox)
        s = self.get_center(stove_bbox)
        return self.distance(p, s)
