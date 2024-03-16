import torch

def intersects(bbox1, bbox2):
    """
    Returns whether two two-dimensional bounding boxes intersect.

    Args:
    - bbox1: Tensor of shape [2, 2] where bbox1[0] gives the xy-coordinate
        of the top-left corner and bbox[1] gives the bottom-right corner.
    - bbox2: Same format as bbox1.

    Returns:
    - A boolean indicating whether the bounding boxes intersect.
    """
    return (bbox1[0, 0] <= bbox2[1, 0] and bbox1[1, 0] >= bbox2[0, 0] and
            bbox1[0, 1] <= bbox2[1, 1] and bbox1[1, 1] >= bbox2[0, 1])

def contains(bbox, p):
    """
    Returns whether a bounding box contains a 2D point p.

    Args:
    - bbox: Tensor of shape [2, 2] where bbox1[0] gives the xy-coordinate
        of the top-left corner and bbox[1] gives the bottom-right corner.
    - p: Tensor of shape [2].

    Returns:
    - A boolean indicating whether bbox contains p.
    """
    return (p[0] <= bbox[1][0] and p[0] >= bbox[0][0] and
            p[1] <= bbox[1][1] and p[1] >= bbox[0][1])

"""
Quadtree data structure to store geometric data with associated bounding boxes.
"""
MAX_DEPTH = 5
class QuadTreeNode:
    def __init__(self, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.data = []
        self.children = []

    def insert(self, bbox, data):
        if len(self.children) != 0:
            for child in self.children:
                if intersects(child.bbox, bbox):
                    child.insert(bbox, data)
        else:
            if len(self.data) != 0 and self.depth < MAX_DEPTH:
                # subdivide
                next_depth = self.depth + 1
                top = self.bbox[0][1]
                left = self.bbox[0][0]
                right = self.bbox[1][0]
                bottom = self.bbox[1][1]

                center = (self.bbox[0] + self.bbox[1]) / 2.
                self.children = [
                    # top-left
                    QuadTreeNode(
                        torch.stack([
                            self.bbox[0],
                            center
                        ]), next_depth),
                    # top-right
                    QuadTreeNode(
                        torch.stack([
                            torch.tensor([center[0], top]),
                            torch.tensor([right, center[1]])
                        ]), next_depth),
                    # bottom-left
                    QuadTreeNode(
                        torch.stack([
                            torch.tensor([left, center[1]]),
                            torch.tensor([center[0], bottom])
                        ]), next_depth),
                    # bottom-right
                    QuadTreeNode(
                        torch.stack([
                            center,
                            self.bbox[1]
                        ]), next_depth),
                ]
                self.data.append((bbox, data))
                for d_bbox, d in self.data:
                    for child in self.children:
                        if intersects(child.bbox, d_bbox):
                            child.insert(d_bbox, d)
                self.data = []
            else:
                self.data.append((bbox, data))

    def leaf_for_point(self, p):
        if not contains(self.bbox, p):
            return None
        for child in self.children:
            l = child.leaf_for_point(p)
            if l:
                return l
        return self
