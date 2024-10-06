# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    if alien.get_shape() == 'Ball':
        center = alien.get_centroid()
        for wall in walls:
            seg = ((wall[0], wall[1]), (wall[2], wall[3]))
            if point_segment_distance(center, seg) <= alien.get_width():
                return True
        return False

    # vertical or horizontal
    head, tail = alien.get_head_and_tail()
    for wall in walls:
        seg = ((wall[0], wall[1]), (wall[2], wall[3]))
        if segment_distance((head, tail), seg) <= alien.get_width():
            return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    x, y = alien.get_centroid()
    rad = alien.get_width()
    if x - rad <= 0 or x + rad >= window[0] or y - rad <= 0 or y + rad >= window[1]:
        return False

    if alien.get_shape() == 'Ball':
        return True

    # vertical or horizontal
    seg = alien.get_head_and_tail()
    width, height = window[0], window[1]
    edges = [
        ((0, 0), (0, height)),  # left
        ((0, 0), (width, 0)),  # top
        ((0, height), (width, height)),  # bottom
        ((width, 0), (width, height))  # right
    ]
    for e in edges:
        if segment_distance(seg, e) <= rad:
            return False
    return True  # if no intersection with boundaries


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """

    def is_collinear(polygon):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = polygon
        return (y2 - y1) * (x3 - x2) == (y3 - y2) * (x2 - x1) and (y3 - y2) * (x4 - x3) == (y4 - y3) * (x3 - x2)

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    if is_collinear(polygon):
        for i in range(4):
            if on_segment(polygon[i], point, polygon[(i + 1) % 4]):
                return True
        return False

    px, py = point
    for i in range(4):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % 4]
        if (y2 - y1) * (px - x1) - (x2 - x1) * (py - y1) > 0:
            return False
    return True

def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    if alien.get_shape() == 'Ball':
        center = alien.get_centroid()
        rad = alien.get_width()
        for wall in walls:
            seg = ((wall[0], wall[1]), (wall[2], wall[3]))
            path = (center, waypoint)
            if segment_distance(path, seg) <= rad:
                return True
        return False

    # vertical or horizontal
    head, tail = alien.get_head_and_tail()
    center = alien.get_centroid()
    head = np.array(head)
    tail = np.array(tail)
    waypoint = np.array(waypoint)
    center = np.array(center)
    head_goal = waypoint + head - center
    tail_goal = waypoint + tail - center
    head_path = (head, head_goal)
    tail_path = (tail, tail_goal)
    begin = (head, tail)
    end = (head_goal, tail_goal)
    for wall in walls:
        wall_seg = ((wall[0], wall[1]), (wall[2], wall[3]))
        if (  # check four segments of the trajectory
                segment_distance(head_path, wall_seg) <= alien.get_width()
                or segment_distance(tail_path, wall_seg) <= alien.get_width()
                or segment_distance(begin, wall_seg) <= alien.get_width()
                or segment_distance(end, wall_seg) <= alien.get_width()
                or is_point_in_polygon((wall[0], wall[1]), (head, tail, tail_goal, head_goal))
                or is_point_in_polygon((wall[2], wall[3]), (head, tail, tail_goal, head_goal))
        ):
            return True
    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    px, py = p
    (x1, y1), (x2, y2) = s

    head_to_point = np.array([px - x1, py - y1])
    segment = np.array([x2 - x1, y2 - y1])

    if np.dot(segment, segment) == 0:
        return np.linalg.norm(head_to_point)

    proj = np.dot(head_to_point, segment) / np.dot(segment, segment)
    proj_point = np.array([x1, y1]) + proj * segment

    if proj < 0:
        closest_point = np.array([x1, y1])
    elif proj > 1:
        closest_point = np.array([x2, y2])
    else:
        closest_point = proj_point

    distance = np.linalg.norm(np.array([px, py]) - closest_point)
    return distance


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    p1, q1 = s1
    p2, q2 = s2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0

    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2

    return min(
        point_segment_distance((x1, y1), s2), point_segment_distance((x2, y2), s2),
        point_segment_distance((x3, y3), s1), point_segment_distance((x4, y4), s1)
    )


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
