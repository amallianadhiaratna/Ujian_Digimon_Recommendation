from flask import Flask, render_template, request, abort, send_from_directory
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objects as go
import chart_studio.plotly as py
import io
import base64 
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


datas=pd.read_json('digimon.json')


app = Flask(__name__)
@app.route('/')
def home():
    return render_template ('home2.html')

@app.route('/hasil', methods=['POST'])
def result():
    if request.method=='POST':
        nama = request.form['nama'].title()
        digimon_user=datas[datas['digimon'].title()==nama][['digimon','stage','type','attribute','image']]
        
        def mergeCol(i):
            return str(i['stage'])+'|'+str(i['type'])+'|'+str(i['attribute'])
        datas['fitur']=datas.apply(mergeCol, axis='columns')

        model=CountVectorizer(tokenizer=lambda x:x.split('|'))
        matrixFeature=model.fit_transform(datas['fitur'])

        score=cosine_similarity(matrixFeature)
        print(score)
        digimon_all=list(enumerate(score[0]))
        sort_digi=sorted(digimon_all, key=lambda i:i[1], reverse=True)

        result=[]

        for i in sort_digi[:7]:
            if datas.iloc[i[0]]['digimon'] != nama:
                digi_suka=datas.iloc[i[0]]['digimon']
                stage_suka=datas.iloc[i[0]]['stage']
                type_suka=datas.iloc[i[0]]['type']
                attrib_suka=datas.iloc[i[0]]['attribute']
                img_suka=datas.iloc[i[0]]['image']
                x={
                    'digi_suka':digi_suka,
                    'stage_suka':stage_suka,
                    'type_suka':type_suka,
                    'attribute_suka':attrib_suka,
                    'img_suka':img_suka
                }
                result.append(x)
                # print(f'{digi_suka} ({stage_suka}) ({type_suka}) {round(i[1]*100)}%')
        return render_template('hasil.html',
                    nama=digimon_user['digimon'].values[0],
                    stage=digimon_user['stage'].values[0], 
                    type=digimon_user['type'].values[0], 
                    attribute=digimon_user['attribute'].values[0], 
                    img=digimon_user['image'].values[0],
                    result=result
                    )

@app.errorhandler(404)
def notFound(error):
    return render_template('error.html')

if __name__ == '__main__':
    app.run(
        debug=True
    )
