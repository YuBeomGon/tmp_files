import json
import numpy as np
import json

from confluent_kafka import Consumer
conf = {'bootstrap.servers': '192.168.0.53:9092',
        'group.id': "NajuPractice",
        'enable.auto.commit': False}
consumer = Consumer(conf)
consumer.subscribe(["transaction"])

with open("pog_local_position.json",'r') as file:
    product_local_pos=json.load(file)
    
cam_4=product_local_pos["cameras"][0]["cabinets"]
#print(cam_4)
cam_7=product_local_pos["cameras"][1]["cabinets"]
cam_t=[cam_4,cam_7]

print(cam_4)

with open('pogmerge.json') as d:
    dictData = json.load(d)
pog_dict = dict(dictData)    

test_array = [[0, 5, 2, "8806011616047", 1],  [1, 2, 3, "8806011616047", 1], [0, 6, 4, "8806011616047", 1], [1, 1, 3, "8806011616047", 2],]

def kafka_pog_msg():
    if np.random.randint(0, 30, 1)[0] == 0 :
        msg = test_array[np.random.randint(0, 30, 1)[0] %4]
        
        return msg
    else :
        msg = consumer.poll(0.00001)
    
    if msg is None:
        return None
    
    msg_py = json.loads(msg.value())
    # received_time = datetime.strptime(msg_py['initial_timestamp'],"%Y-%m-%d-%H:%M:%S.%f") #time_info
    shelf_num = str(msg_py['shelf_id'])
    shelf_row = int(msg_py['row'])
    col = int(msg_py['column'])
    barcode = msg_py['barcode']
    qty = msg_py['qty']
    p_name = msg_py['name']
    
    if str(barcode) in pog_dict.keys() :
        return shelf_num, shelf_row, col, pog_dict[str(barcode)], qty
    else :
        print('no matching pog')
        return None

def pd_coord(cam_num, sh_num, row,col):
    row -= 1
    col -= 1
    list_result=[k["annotation"] for k in cam_t[cam_num] if k["shelfNum"]==sh_num]
    print(list_result)
    #list_result=np.array(list_result)
    #list_result.shape
    #print(list_result)
    if list_result:
        list_result=list_result[0]
        print(list_result[row][row])
        x=list_result[row][col*3:(col+1)*3][0]
        y=list_result[row][col*3:(col+1)*3][1]
        return x,y
    else:
        return None
    
def jsontodict(file) :
    with open(file) as d:
        dictData = json.load(d)