from asyncio import subprocess
from flask import Flask
from flask import request
from flask import jsonify
import datetime
import os
import pickle
import numpy as np
# import utils
# import hit
import keras

startTime = datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S")

app = Flask(__name__)

# model path
mpath = "./dummy/tf-2022-0615-wisata.pckl"
mpath = "./dummy/tf-2022-0615-wisata.pckl"
# mo1 = pickle.load(open(mpath, "rb"))

mo1 = keras.models.load_model("./dummy/tf-2022-0615-wisata.mdl")


mainLabel = [
{"no":1,"data":"0000000","lbl":"1000000000000000000","label_tag":"'1000000000000000000"},
{"no":2,"data":"0000010","lbl":"0100000000000000000","label_tag":"'0100000000000000000"},
{"no":3,"data":"0000100","lbl":"0010000000000000000","label_tag":"'0010000000000000000"},
{"no":4,"data":"0001000","lbl":"0001000000000000000","label_tag":"'0001000000000000000"},
{"no":5,"data":"0100000","lbl":"0000100000000000000","label_tag":"'0000100000000000000"},
{"no":6,"data":"0100100","lbl":"0000010000000000000","label_tag":"'0000010000000000000"},
{"no":7,"data":"0101000","lbl":"0000001000000000000","label_tag":"'0000001000000000000"},
{"no":8,"data":"0110000","lbl":"0000000100000000000","label_tag":"'0000000100000000000"},
{"no":9,"data":"0111000","lbl":"0000000010000000000","label_tag":"'0000000010000000000"},
{"no":10,"data":"1001000","lbl":"0000000001000000000","label_tag":"'0000000001000000000"},
{"no":11,"data":"1001100","lbl":"0000000000100000000","label_tag":"'0000000000100000000"},
{"no":12,"data":"1010010","lbl":"0000000000010000000","label_tag":"'0000000000010000000"},
{"no":13,"data":"1100000","lbl":"0000000000001000000","label_tag":"'0000000000001000000"},
{"no":14,"data":"1101000","lbl":"0000000000000100000","label_tag":"'0000000000000100000"},
{"no":15,"data":"1101001","lbl":"0000000000000010000","label_tag":"'0000000000000010000"},
{"no":16,"data":"1110000","lbl":"0000000000000001000","label_tag":"'0000000000000001000"},
{"no":17,"data":"1110001","lbl":"0000000000000000100","label_tag":"'0000000000000000100"},
{"no":18,"data":"1111000","lbl":"0000000000000000010","label_tag":"'0000000000000000010"},
{"no":19,"data":"1111001","lbl":"0000000000000000001","label_tag":"'0000000000000000001"}
]
hobi = [
    {"label":"Bekerja","id":1},
    {"label":"Belajar","id":2},
    {"label":"Belanja","id":3},
    {"label":"Bermain","id":4},
    {"label":"Dakwah","id":5},
    {"label":"Fotografi","id":6},
    {"label":"Jualan","id":7},
    {"label":"Kuliner","id":8},
    {"label":"Main Game","id":9},
    {"label":"Memancing","id":10},
    {"label":"Membaca","id":11},
    {"label":"Menggambar","id":12},
    {"label":"Menjahit","id":13},
    {"label":"Menonton","id":14},
    {"label":"Menulis","id":15},
    {"label":"Merawat Anak","id":16},
    {"label":"Musik","id":17},
    {"label":"Ngopi","id":18},
    {"label":"Olahraga","id":19},
    {"label":"Otomotif","id":20},
    {"label":"Politikus","id":21},
    {"label":"Seni","id":22},
    {"label":"Tidur","id":23},
    {"label":"Traveling","id":24}
]
status = [
    {"label":"Belum nikah","id":1},
    {"label":"Nikah","id":2}
]
mainPG = [ "panah", "labirin", "paintball", "ffox", "tamankelinci", "atv", "sepeda" ]
# mainLabel, mainPG

def parseGames(no):
    no = int(no)
    ml1 = [m for m in mainLabel if m["no"]==no]
    
    return [pg for pg,ml in zip(mainPG, list(ml1[0]["data"]) ) if ml=="1"]

def stdJson(success, msg, data):
    return {
        "success": True if success else False,
        "message": msg,
        "data": data,
    }

def success(msg, data):
    return stdJson(True, msg, data)

def fail(msg, data):
    return stdJson(False, msg, data), 400

@app.route("/")
def home() :
    global startTime
    return "run"

@app.route("/json")
def test() :
    return {
        "key": "v",
        "key2": "v2",
    }

"""
umur=19&status=1&rombongan=2&hobi=11
"""
@app.route("/predict")
def predict():
    data = {}
    try:
        row = [
            request.args.get('umur'),
            request.args.get('status'),
            request.args.get('rombongan'),
            request.args.get('hobi'),
        ]
        defs = [19,1,2,11]
        if row.count(None) > 0:
            row = defs
        
        row = [int(x) for x in row]

        # 19,1,2,11
        yhat = mo1.predict([row])
        no = np.argmax(yhat)
        # data = [row, "class", float(no), parseGames(int(no))]

        return success("Ok", {
            "input": row,
            "class": float(no),
            "parsed": parseGames(int(no)),
        })
    except Exception as e:
        return fail("fail "+e, data)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0')
