import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import ros_numpy 
import ins2ref_vel_cluster as ins2ref
import numpy as np

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('intensity', 12, PointField.FLOAT32, 1)]
header = Header()
header.frame_id = "crl_rzr/sensor_origin_link"

def scan_callback(data):
    points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(data)
    #print(np.max(points[:,3]))
    new_points = ins2ref.ref_conv(points)
    #print(new_points.dtype.names)
    header.stamp = rospy.Time.now()
    points_msg = pc2.create_cloud(header, fields, new_points)
    scan_pub.publish(points_msg)
cal_node = rospy.init_node('cal_node',anonymous =True)
scan_pub = rospy.Publisher('/calibrated_laserscan',PointCloud2,queue_size = 10)
rospy.Rate(10)

while not rospy.is_shutdown():
    rospy.Subscriber('/crl_rzr/velodyne_points_agg',PointCloud2,scan_callback,queue_size=1)
    rospy.spin()
