category2obj = {}
datasets = ['pybullet', 'ycb', 'housecat']
for d in datasets:
    with open('object_%s.csv'%d, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.replace('\n', '').split(',')
            category = elements[0]
            objects = ['%s/%s'%(d,e) for e in elements[1:] if e!='']
            if category not in category2obj:
                category2obj[category] = objects
            else:
                category2obj[category] += objects
print(category2obj)

with open('categories_dict.txt', 'w') as g:
    g.writelines("{\n")
    for k in sorted(category2obj.keys()):
        g.writelines("\t'%s': "%k)
        g.writelines(str(category2obj[k]))
        g.writelines(',\n')
    g.writelines("}")

