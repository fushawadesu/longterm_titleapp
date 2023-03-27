import os
from flask import Flask, flash, request, redirect, url_for, render_template, make_response
from werkzeug.utils import secure_filename
from flask import render_template_string

from transformers import AutoModel , AutoTokenizer
import torch
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import re
from bs4 import BeautifulSoup

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['xml'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

df = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global df
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = (os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #空のデータフレーム作成
            col_name = ["filename", "tagname", "text"]
            df = pd.DataFrame(columns=col_name)


            
            with open(filepath) as f:
                soup = BeautifulSoup(f, 'xml')

                #xmlファイルをstr変換
                pend_basic = str(soup)

                #不要な文字列やタグの削除
                pattern = r'<xref.*?>'
                pend = re.sub(pattern, '', pend_basic)
                pend = re.sub(r'<italic>', '', pend)
                pend = re.sub(r'</italic>', '', pend)
                pend = re.sub('<kwd>','',pend)
                pend = re.sub('</kwd>','',pend)
                
                #<title>ごとに区切る
                _text = pend.split("<title>")
                #区切りの長さを確認する
                len(_text)
                #text[0]はゴミなので不要。１からのリストに変更
                _text = _text[1:len(_text)]
                #</titleの不要な箇所を削除>
                text = []
                for v in _text:
                    index = v.find("</title>")
                    if index != -1:
                        v = v[index + len("</title>"):]
                        text.append(v)
                    else:
                        text.append(v)

                #tagnameのリストを作成
                tagname = [tag.text for tag in soup('title')]

                #filenameのを列分だけ追加
                filename = []
                for i in tagname:
                    filename.append(filepath[9:])
                
                #zippedfileの作成
                zippedList =  list(zip(filename, tagname, text))

                #dataframe化する
                _df = pd.DataFrame(zippedList, columns = ["filename", "tagname","text"])

                #dataframeに追加する
                df = df.append(_df, ignore_index=True)

                #モデルとトークナイザーを指定
                tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", use_fast=False)
                model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
                #機械学習を行うため、cpuとデバイスの指定
                cuda_id = "cuda:" + str(0)
                device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

                #listの作成
                title_list = ['Introduction', 'Method', 'Result', 'Conclusion', 'Acknowledgements', 'Reference']

                #titlelistのトークナイザーを作成
                #トークン化するテキスト、パディングを行う方法、トークンの最大長、トークン化後の結果をTensorオブジェクトとして返すよう指示するreturn_tensors="pt"引数が含まれている
                #下記スクリプトを実行することで、自然言語処理を行うことができる
                inputs = tokenizer.batch_encode_plus(
                    title_list,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                #モデルのデバイス設定
                model.to(device)
                model.eval()

                with torch.no_grad():
                    outputs = model(**inputs.to(device))
                word_embeddings = np.array(outputs.hidden_states[12].cpu(), np.float32)
                words_num = word_embeddings.shape[0]
                title_vec_list = []
                for i in range(words_num):
                    token_length = int(torch.count_nonzero(inputs["attention_mask"][i]))
                    word_vec = word_embeddings[i][0:token_length]
                    word_vec = list(np.mean(word_vec, axis=0))
                    title_vec_list.append(word_vec)
                
                def word_to_vec(word):
                    inputs = tokenizer.batch_encode_plus(
                        [word],
                        padding="max_length",
                        max_length=128,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(**inputs.to(device))
                    word_embeddings = np.array(outputs.hidden_states[12].cpu(), np.float32)
                    words_num = word_embeddings.shape[0]
                    title_vec_list = []
                    for i in range(words_num):
                        token_length = int(torch.count_nonzero(inputs["attention_mask"][i]))
                        word_vec = word_embeddings[i][0:token_length]
                        word_vec = list(np.mean(word_vec, axis=0))
                    return word_vec
                
                title_vecs = [word_to_vec(title) for title in title_list]

                def cos_sim(v1, v2):
                    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                for i in range(0, len(title_list)):
                    df[f'{title_list[i]}'] = df['tagname'].apply(lambda x: cos_sim(word_to_vec(x), title_vecs[i]))
                
                # 最大値を取る列=title listを指定する
                cols_to_max = title_list

                # 行ごとに最大値を取り、新しい列として追加する
                df['max_column'] = df[cols_to_max].idxmax(axis=1)
                result = df.to_html(index=False)
        return render_template('result.html', table=result)
    return render_template('index.html')

@app.route('/download')
def download():
    global df
    if df is None:
        return "No data"
    response = make_response(df.to_csv(index=False))
    response.headers['Content-Disposition'] = 'attachment; filename=result.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response


from flask import send_from_directory
@app.route('/uploads/<name>')
def uploaded_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == '__main__':
    app.run(debug=True)