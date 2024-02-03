import os

split = 'test'
categories = ['_'.join(c.split('_')[:-1]) for c in os.listdir() if os.path.isdir(c) and c.endswith(split)]
category2obj = {}
num_objects = 0
for cat in categories:
    category2obj[cat] = []
    objects = [f for f in os.listdir(cat + '_' + split) if f.endswith('.png')]
    for obj in objects:
        dataset = obj.split('_')[0]
        object_name = '_'.join(obj.replace('.png', '').split('_')[1:])
        object_name = dataset + '/' + object_name
        category2obj[cat].append(object_name)
        num_objects += 1

print(category2obj)
print('Total number of objects:', num_objects)

with open('categories_dict_test.txt', 'w') as g:
    g.writelines("{\n")
    for k in sorted(category2obj.keys()):
        g.writelines("\t'%s': "%k)
        g.writelines(str(category2obj[k]))
        g.writelines(',\n')
    g.writelines("}")

