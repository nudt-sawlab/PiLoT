import math
sensor_height = 10.56
f_mm = 5.5
fovy_radians = 2 * math.atan(sensor_height / 2 / f_mm)
fovy_degrees = math.degrees(fovy_radians)
print(fovy_degrees)