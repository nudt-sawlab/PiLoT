import math
# 假设你的输入文本叫 poses.txt
input_txt = "/mnt/sda/MapScape/pose/Switzerland_seq3.txt"
output_kml = "/mnt/sda/MapScape/pose/Switzerland_seq3.kml"

# KML文件头、尾固定写好
kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <gx:Tour>
    <name>My SixDOF Tour</name>
    <gx:Playlist>
"""

kml_footer = """    </gx:Playlist>
  </gx:Tour>
</Document>
</kml>
"""

flyto_list = []  # 用来存放每个 <gx:FlyTo> 片段的字符串

with open(input_txt, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        # 假设每行: filename heading pitch roll lon lat alt
        parts = line.split()
        if len(parts) < 7:
            continue
        
        filename = parts[0]   # 例如 251.jpg
        heading = float(parts[5])        # roll
        tilt  = float(parts[4])          # yaw
        roll   = float(parts[6])   # ptich
        lon     = float(parts[1])
        lat     = float(parts[2])
        alt     = float(parts[3])
        
        # Google Earth: tilt=0 -> 垂直向下, tilt=90 -> 水平
        # 如果你的 pitch 是 "飞机机头上扬" 为正，且 0=水平，则可能需要 tilt = 90 - pitch
        # 这里随意举例:
        # tilt = 90.0 - pitch
        
        # heading 在 Google Earth 里 0=北, 90=东，如果你的 heading 定义一致则可直接用
        # roll 在 Google Earth 中 -180 ~ 180，若你的 roll 定义一致则可直接用
        
        gx_flyto = f"""      <gx:FlyTo>
        <gx:duration>1</gx:duration>
        <Camera>
          <longitude>{lon}</longitude>
          <latitude>{lat}</latitude>
          <altitude>{alt}</altitude>
          <heading>{heading}</heading>  
          <tilt>{0}</tilt>
          <roll>{roll}</roll>
          <altitudeMode>absolute</altitudeMode>
        </Camera>
      </gx:FlyTo>
"""
        flyto_list.append(gx_flyto)

# 最终写出KML
with open(output_kml, 'w', encoding='utf-8') as f:
    f.write(kml_header)
    for flyto_block in flyto_list:
        f.write(flyto_block)
    f.write(kml_footer)

print("KML文件已生成:", output_kml)
