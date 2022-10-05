import os
import uuid
import datetime

file_list=os.listdir('./')
for i in file_list:
    file_uuid=uuid.uuid4()
    file_uuid=str(file_uuid)
    date=datetime.datetime.now().strftime("%y%m%d")
    base="showroom"
    print(i)
    print("-"*10)
    new_name="_".join([base,date,file_uuid])
    print(new_name+'.jpg')
    os.rename(i,new_name+'.jpg')

