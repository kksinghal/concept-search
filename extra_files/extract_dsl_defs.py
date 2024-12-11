f = open("dsl.txt", "r")
data = f.read()
data
defs = []
idx = data.find("def")
while idx != -1:
    i = data.find('"""')
    i = data[i+3:].find('"""') + i+3
    defs.append(data[idx:i+3])

    data = data[i+3:]
    idx = data.find("def")

f2 = open("dsl_defs.txt", "w")
f2.write("\n".join(defs))