import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from typing import NamedTuple, List, Tuple
import numpy as np

EPSILON = 0.001

class Point():
    x: float
    y: float

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def as_tuple(self) -> (float, float):
        return (self.x, self.y)

    def as_array(self) -> np.array:
        return np.array([self.x, self.y]).astype(float)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        
        return False

    @staticmethod
    def to_tuple(point) -> (float, float):
        return (point.x, point.y)

    @staticmethod
    def list_to_tuple(vert_list: list):
        return list(
            map(
                lambda x: Point.to_tuple(x), vert_list))

Vertices_List = List[Point]
Vertices_Orientations_List = Tuple[Point, int]
 
class Geometry():
    @staticmethod
    def get_segments_intersection_point(a1: np.array, a2: np.array
        , b1: np.array, b2: np.array) -> np.array:

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = Geometry.perp(da, db)

        if abs(dap) < EPSILON:
            return None

        si = Geometry.perp(db, dp) / dap

        if si < 0 or si > 1:
            return None
        
        ti = Geometry.perp(da, dp) / dap

        if ti < 0  or ti > 1:
            return None

        return a1 + si * da

    @staticmethod
    def get_segment_len(a1: np.array, a2: np.array) -> float:
        dx2 = (a1[0] - a2[0]) ** 2
        dy2 = (a1[1] - a2[1]) ** 2

        return pow(dx2 + dy2, 1/2)

    @staticmethod
    def is_point_inside_polygon(
        polygon_vertices: Vertices_List
        , point: Point) -> bool:
        
        x_min = min(polygon_vertices, key=lambda p: p.x).x
        x_max = max(polygon_vertices, key=lambda p: p.x).x
        y_min = min(polygon_vertices, key=lambda p: p.y).y
        y_max = max(polygon_vertices, key=lambda p: p.y).y

        bounding_box = [Point(x_min, y_min), Point(x_max, y_max)]
        is_inside_bounding_box = False

        if bounding_box[0].x < point.x < bounding_box[1].x and bounding_box[0].y < point.y < bounding_box[1].y:
            is_inside_bounding_box = True

        if not is_inside_bounding_box:
            return False

        #  http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
        inside = False
        j = len(polygon_vertices) - 1
        for idx, vert in enumerate(polygon_vertices):
            if (vert.y > point.y) != (polygon_vertices[j].y > point.y) and \
                (point.x < (polygon_vertices[j].x - vert.x * (point.y-vert.y) \
                    / (polygon_vertices[j].y - vert.y) + vert.x)):

                inside = not inside
            j = idx
        
        return inside

    @staticmethod
    def perp(a, b) -> float:
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def get_segment_in_y_bounds_len(
        a1: Point
        , a2: Point
        , bound_y_min: float
        , bound_y_max: float
        , box_x_min: float = -100
        , box_x_max: float = 100) -> float:

        segment_min = min([a1, a2], key=lambda x: x.y)
        segment_max = max([a1, a2], key=lambda x: x.y)
        segment_min_in_bounds: Point = None
        segment_max_in_bounds: Point = None

        # segment lies on line
        if segment_max.y == segment_min.y and \
                (segment_max.y == bound_y_max or segment_max.y == bound_y_min):

            return Geometry.get_segment_len(a1.as_array(), a2.as_array())

        if segment_max.y <= bound_y_min or segment_min.y >= bound_y_max:
            return 0

        if segment_min.y >= bound_y_min:
            segment_min_in_bounds = segment_min.as_array()
        else:
            segment_min_in_bounds = Geometry.get_segments_intersection_point(
                a1.as_array(),
                a2.as_array()
                , np.array([box_x_min, bound_y_min])
                , np.array([box_x_max, bound_y_min]))
        
        if segment_max.y <= bound_y_max:
            segment_max_in_bounds = segment_max.as_array()
        else:
            segment_max_in_bounds = Geometry.get_segments_intersection_point(a1.as_array()
                , a2.as_array()
                , np.array([box_x_min, bound_y_max])
                , np.array([box_x_max, bound_y_max]))
        
        return Geometry.get_segment_len(segment_min_in_bounds, segment_max_in_bounds)

class Orient():
    # left
    L = 1
    # straight
    S = 0
    # right
    R = -1     

class Polygon():
    vertices: Vertices_List = list()
    vertices_and_kernel_points: Vertices_List = list()
    vertices_orientations: Vertices_Orientations_List = list()
    upper_spike_list: Vertices_List = list()
    lower_spike_list: Vertices_List = list()
    min_lower_spike_y: float = None
    max_upper_spike_y: float = None
    upper_kernel_bound: float = None
    lower_kernel_bound: float = None
    bounding_box: Tuple[Point, Point] = tuple()
    kernel_exists: bool = False
    kernel_intersection_list: Vertices_List = list()
    offset = 0

    def __init__(self, vertices: Vertices_List):
        self.vertices = vertices
    
    def draw_polygon(self, autoclose=True) -> None:
        if autoclose and self.vertices[0] != self.vertices[-1]:
            self.vertices.append(self.vertices[0])
        
        path = Path(Point.list_to_tuple(self.vertices), closed=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
        
        if self.kernel_exists:
            ax.axhline(y=self.upper_kernel_bound)
            ax.axhline(y=self.lower_kernel_bound)

        ax.set_xlim(self.bounding_box[0].x-1, self.bounding_box[1].x+1)
        ax.set_ylim(self.bounding_box[0].y-1, self.bounding_box[1].y+1)
        
        plt.show()

    def assign_orientation(self) -> None:
        orientation_list = list()
        for idx, elem in enumerate(self.vertices):
            line_y = self.vertices[idx-1].y

            point_array = np.array(
                [[0, line_y, 1]
                , [1, line_y, 1]
                , [elem.x, elem.y, 1]])
            
            det = np.linalg.det(point_array)
            
            if det > EPSILON:
                orientation = Orient.L
            elif -EPSILON <= det <= EPSILON:
                orientation = Orient.S
            elif det < EPSILON:
                orientation = Orient.R

            orientation_list.append(orientation)

            # print("idx: %x, orient %s" % (idx,  orientation))

        self.vertices_orientations = list(zip(self.vertices, orientation_list))

    def get_spikes(self):
        for idx, vert_orient in enumerate(self.vertices_orientations):
            vert, orient = vert_orient
            next_orient = self.vertices_orientations[(idx+1) % len(self.vertices_orientations)][1]
            if next_orient == Orient.S:
                next_orient = self.vertices_orientations[(idx+2) % len(self.vertices_orientations)][1]
            # if (orient == 'L' and next_orient =='R') or (orient == 'R' and next_orient =='L'):
            if orient == -next_orient:
                cpy = [v for v, o in self.vertices_orientations]
                del cpy[idx]

                if Geometry.is_point_inside_polygon(cpy, vert):
                    if orient == Orient.L:
                        self.upper_spike_list.append(vert)
                    elif orient == Orient.R:
                        self.lower_spike_list.append(vert)

    def get_polygon_kernel_y_range(self) -> None:
        self.assign_orientation()
        self.get_spikes()
        
        if self.upper_spike_list:
            self.max_upper_spike_y = max(self.upper_spike_list, key=lambda p: p.y).y
            
        if self.lower_spike_list:
            self.min_lower_spike_y = min(self.lower_spike_list, key=lambda p: p.y).y

        self.lower_kernel_bound = self.max_upper_spike_y \
            if self.upper_spike_list \
            else min(self.vertices, key=lambda po: po.y).y
        
        self.upper_kernel_bound = self.min_lower_spike_y \
            if self.lower_spike_list \
            else max(self.vertices, key=lambda po: po.y).y

    def get_polygon_crossing_point(self, axis_y: float):
        # intersections_list: List[Point] = list()
        # self.vertices_and_kernel_points = self.vertices[:]

        for idx, vert in enumerate(self.vertices):
            next_vert = self.vertices[(idx+1) % len(self.vertices)].as_array()
            potential_intersection = Geometry.get_segments_intersection_point(vert.as_array(), next_vert
                , np.array([self.bounding_box[0].x-1, axis_y]), np.array([self.bounding_box[1].x+1, axis_y]))
            
            if potential_intersection is not None:
                self.kernel_intersection_list.append(potential_intersection)
                potential_intersection_point = Point(potential_intersection[0], potential_intersection[1])
                if (potential_intersection_point not in self.vertices_and_kernel_points):
                    insert_idx = self.vertices_and_kernel_points.index(vert) + 1
                    self.vertices_and_kernel_points.insert(insert_idx, Point(potential_intersection[0], potential_intersection[1]))
            
            # print("{} {} {}".format(vert.as_array(), next_vert, potential_intersection))
        
        return (min(self.kernel_intersection_list, key=lambda x: x[0])
            , max(self.kernel_intersection_list, key=lambda x: x[0]))

    def get_bound_len(self, upper: bool) -> float:
        bound_y: float = None
        
        if upper:
            if not self.lower_spike_list:
                return 0
            bound_y = self.min_lower_spike_y
        
        else:
            if not self.upper_spike_list:
                return 0
            bound_y = self.max_upper_spike_y

        left_right_bound_lim = self.get_polygon_crossing_point(bound_y)
        return abs(left_right_bound_lim[1][0] - left_right_bound_lim[0][0])
        
    def get_sides_len(self) -> float:
        sides_len: float = 0
        for idx, cur_vert in enumerate(self.vertices):
            next_vert: Point = self.vertices[(idx+1)%len(self.vertices)]
            
            sides_len += Geometry.get_segment_in_y_bounds_len(
                cur_vert
                , next_vert
                , self.lower_kernel_bound
                , self.upper_kernel_bound
                , self.bounding_box[0].x
                , self.bounding_box[1].x)

        return sides_len

    def get_kernel_circumference(self) -> float:
        if not self.kernel_exists:
            return 0

        self.vertices_and_kernel_points = self.vertices[:]
        uppper_bound_len = self.get_bound_len(upper=True)
        lower_bound_len = self.get_bound_len(upper=False)
        sides_len = self.get_sides_len()
        # print("u {} l {} s {}".format(uppper_bound_len, lower_bound_len, sides_len))
        return uppper_bound_len + lower_bound_len + sides_len

    def get_bounding_box(self) -> [Point, Point]:
        min_x = min(self.vertices, key=lambda x: x.x).x
        min_y = min(self.vertices, key=lambda x: x.y).y
        max_x = max(self.vertices, key=lambda x: x.x).x
        max_y = max(self.vertices, key=lambda x: x.y).y

        self.bounding_box = [Point(min_x, min_y), Point(max_x, max_y)]

        return self.bounding_box

    def kernel_exists_msg(self) -> None:
        if self.lower_kernel_bound >= self.upper_kernel_bound:
            self.kernel_exists = False
            print("Kernel does not exist")
        else:
            self.kernel_exists = True
            print("Kernel exists")

    def get_vertices_inside_kernel(self):
        def is_in_kernel(v):
            return self.lower_kernel_bound <= v.y <= self.upper_kernel_bound

        vertices_inside_kernel_filter = filter(is_in_kernel, self.vertices_and_kernel_points)
        vertices_inside_kernel_list = list(vertices_inside_kernel_filter)
        return vertices_inside_kernel_list

    def get_kernel_area(self):
        kernel_vertices = self.get_vertices_inside_kernel()
        area = 0
        # for idx, vert in enumerate(kernel_vertices):
        #     next_vert = kernel_vertices[(idx+1) % len(kernel_vertices)]
        #     area += vert.x * next_vert.y - vert.y * next_vert.x

        for idx, vert in enumerate(kernel_vertices):
            next_vert = kernel_vertices[(idx+1) % len(kernel_vertices)]
            prev_vert = kernel_vertices[(idx-1) % len(kernel_vertices)]
            area += (next_vert.x - prev_vert.x) * vert.y

        return -area / 2

    def run(self):
        poly.get_bounding_box()
        poly.get_polygon_kernel_y_range()
        poly.kernel_exists_msg()
        circumference = poly.get_kernel_circumference()
        print("Kernel circumference: {}".format(circumference))
        area = poly.get_kernel_area()
        print("Kernel area: {}".format(area))
        poly.draw_polygon()

# polygon_vertices: Vertices_List = [
#     Point(-5., -5.)
#     , Point(5., -5.)
#     , Point(5., 5.)
#     , Point(0., 5.)
#     , Point(-1., 1.)
#     , Point(-4., 5.)
# ]  

# polygon_vertices: Vertices_List = [
#     Point(0, 0)
#     , Point(2, 2)
#     , Point(3, 1)
#     , Point(5, 3)
#     , Point(7, 4)
#     , Point(5, 5)
#     , Point(3, 8)
#     , Point(2, 6)
#     , Point(-5, 8)
#     , Point(-3, 6)
#     , Point(-6, 4)
# ]  

# polygon_vertices: Vertices_List = [
#     Point(0, 0)
#     , Point(5, 0)
#     , Point(7, 8)
#     , Point(9, 0)
#     , Point(11, 0)
#     , Point(11, 10)
#     , Point(4, 10)
#     , Point(2, 5)
#     , Point(0, 10)
# ]  

# polygon_vertices: Vertices_List = [
#     Point(0, 0)
#     , Point(3, 0)
#     , Point(3, 2)
#     , Point(5, 2)
#     , Point(5, 0)
#     , Point(7, 0)
#     , Point(7, 5)
#     , Point(5, 5)
#     , Point(3, 8)
#     , Point(1, 5)
#     , Point(0, 5)
# ]  


######

# polygon_vertices: Vertices_List = [
#     Point(-6, -8)
#     , Point(-1, -5)
#     , Point(-3, -3)
#     , Point(1, 1)
#     , Point(-4, 5)
#     , Point(-1, 1)
#     , Point(-5, -4)
#     , Point(-4, -5)
# ]

# polygon_vertices: Vertices_List = [
#     Point(-6, -5)
#     , Point(-2, 1)
#     , Point(-1, -4)
#     , Point(1, 2)
#     , Point(3, -5)
#     , Point(3, 6)
#     , Point(-2, 6)
#     , Point(-2, 1)
#     , Point(-4, 1)
#     , Point(-4, 5)
#     , Point(-7, 5)
# ]

# polygon_vertices: Vertices_List = [
#     Point(-4, -4)
#     , Point(1, 1)
#     , Point(2, -3)
#     , Point(2, 4)
#     , Point(-6, 6)
# ]  

polygon_vertices: Vertices_List = [
    Point(-7, -2)
    , Point(-3, -2)
    , Point(0, 2)
    , Point(1, -2)
    , Point(3, -2)
    , Point(2, 2)
    , Point(4, 2)
    , Point(4, -1)
    , Point(9, -1)
    , Point(9, 3)
    , Point(8, 3)
    , Point(8, 8)
    , Point(4, 8)
    , Point(2, 7)
    , Point(0, 8)
    , Point(-1, 7)
    , Point(-3, 7)
    , Point(-6, 8)
    , Point(-7, 7)
    , Point(-4, 6)
    , Point(-4, 5)
    , Point(-6, 4)
    , Point(-3, 3)
    , Point(-6, 3)
]  

poly = Polygon(polygon_vertices)
poly.run()