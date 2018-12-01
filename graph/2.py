temp=[]
import re
with open(r"C:\Users\james\Desktop\graphwave\graph\labels-usa-airports.txt",'r') as inp:
    inp.readline()
    for i in inp:
        nums=i.split()
        temp.append((int(nums[0]),int(nums[1])))

temp.sort(key=lambda x:x[0])


with open(r"C:\Users\james\Desktop\graphwave\graph\labels-usa-airports1.txt",'w') as out:
    for i in temp:
        out.write(str(i[0])+" "+str(i[1])+'\n')