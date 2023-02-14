import tools

names = tools.getClassNames("ssd_mobilenet_v3_files/coco.names")

print(tools.getClassNameFromId(names, 53))