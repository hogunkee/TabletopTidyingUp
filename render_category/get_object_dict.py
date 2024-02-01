import os

categories = [c for c in os.listdir() if os.path.isdir(c)]
category2obj = {}
for cat in categories:
    category2obj[cat] = []
    objects = [f for f in os.listdir(cat) if f.endswith('.png')]
    for obj in objects:
        dataset = obj.split('_')[0]
        object_name = '_'.join(obj.replace('.png', '').split('_')[1:])
        category2obj[cat].append(object_name)

print(category2obj)

with open('categories_dict.txt', 'w') as g:
    g.writelines("{\n")
    for k in sorted(category2obj.keys()):
        g.writelines("\t'%s': "%k)
        g.writelines(str(category2obj[k]))
        g.writelines(',\n')
    g.writelines("}")

