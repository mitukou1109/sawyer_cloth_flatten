#!/usr/bin/env python

import cv2 as cv

import rospy

from cv_bridge import CvBridge
from geometry_msgs import msg as geometry_msgs
from sensor_msgs import msg as sensor_msgs
import tf2_geometry_msgs
import tf2_ros

from sawyer_cloth_flatten.srv import Get3DPointFromPixel, GetHighestPositionOfCloth

class RealSenseHandler:
    def __init__(self):
        self._cloth_contour_image_pub = rospy.Publisher('cloth_contour/image', sensor_msgs.Image, queue_size=10)
        self._camera_info_sub = rospy.Subscriber('/camera/color/camera_info', sensor_msgs.CameraInfo, self._camera_info_cb)
        self._color_image_sub_ = rospy.Subscriber('/camera/color/image_raw', sensor_msgs.Image, self._color_image_cb)
        self._depth_image_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', sensor_msgs.Image, self._depth_image_cb)

        self._get_3d_point_from_pixel_service_server = rospy.Service(
            'get_3d_point_from_pixel', Get3DPointFromPixel, self._get_3d_point_from_pixel_cb)
        self._get_highest_position_of_cloth_service_server = rospy.Service(
            'get_highest_position_of_cloth', GetHighestPositionOfCloth, self._get_highest_position_of_cloth_cb)

        self._cv_bridge = CvBridge()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

    def _camera_info_cb(self, camera_info):
        self._intrinsics = camera_info.K

    def _color_image_cb(self, color_image):
        self._color_image = color_image

    def _depth_image_cb(self, depth_image):
        self._depth_image = depth_image

    def _get_3d_point_from_pixel_cb(self, request):
        return self._get_3d_point_from_pixel(request.pixel_x, request.pixel_y)

    def _get_highest_position_of_cloth_cb(self, request):
        return self.get_highest_position_of_cloth()

    def get_highest_position_of_cloth(self):
        orig_image = self._cv_bridge.imgmsg_to_cv2(self._color_image, 'rgb8')

        hsv_image = cv.cvtColor(orig_image, cv.COLOR_RGB2HSV)
        _, _, v_image = cv.split(hsv_image)
        v_image = cv.blur(v_image, (9, 9))
        # _, thresh = cv.threshold(v_image, 0, 255, cv.THRESH_OTSU)
        _, thresh = cv.threshold(v_image, 150, 255, cv.THRESH_BINARY_INV)

        _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if not contours:
            rospy.logwarn('Failed to detect cloth')
            return None

        cloth_contour = max(contours, key=cv.contourArea)

        depth_image = self._cv_bridge.imgmsg_to_cv2(self._depth_image).copy()
        highest_pixel_of_cloth = None
        while highest_pixel_of_cloth is None:
            highest_val, _, highest_pixel, _ = cv.minMaxLoc(depth_image)
            print(highest_val, highest_pixel)
            if highest_val >= 5 and cv.pointPolygonTest(cloth_contour, highest_pixel, measureDist=False) >= 0:
                highest_pixel_of_cloth = highest_pixel
            else:
                depth_image[tuple(reversed(highest_pixel))] = 2000

        cloth_contour_image = orig_image.copy()
        cv.polylines(cloth_contour_image, cloth_contour, isClosed=True, color=(0, 0, 255), thickness=5)
        cv.circle(cloth_contour_image, highest_pixel_of_cloth, radius=5, color=(255, 0, 0), thickness=-1)

        cloth_contour_image_msg = self._cv_bridge.cv2_to_imgmsg(cloth_contour_image, self._color_image.encoding)
        cloth_contour_image_msg.header.stamp = rospy.Time.now()
        cloth_contour_image_msg.header.frame_id = self._color_image.header.frame_id
        self._cloth_contour_image_pub.publish(cloth_contour_image_msg)

        return self._get_3d_point_from_pixel(*highest_pixel_of_cloth)

    def _get_3d_point_from_pixel(self, x, y):
        depth_image = self._cv_bridge.imgmsg_to_cv2(self._depth_image)
        point_z = depth_image[int(y), int(x)] / 1E3
        point = geometry_msgs.Point(x=(x-self._intrinsics[2])*point_z/self._intrinsics[0],
                                    y=(y-self._intrinsics[5])*point_z/self._intrinsics[4],
                                    z=point_z)

        rate = rospy.Rate(10.0)
        transform = None
        while transform is None:
            try:
                transform = self._tf_buffer.lookup_transform('base', 'camera_depth_optical_frame', rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn('%s', e.what())
                rate.sleep()
                continue

        return tf2_geometry_msgs.do_transform_point(geometry_msgs.PointStamped(point=point), transform)

def main():
    rospy.init_node('realsense_handler')

    realsense_handler = RealSenseHandler()

    rospy.spin()

if __name__ == '__main__':
    main()