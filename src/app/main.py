from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle
import pandas as pd
import os

####### Treino e Retreino do modelo ########

#df = pd.read_csv('../../data/processed/casas.csv')
# Modelo apenas com o tamanho
#X = df['tamanho']
#y = df['preco']

#X = df.drop('preco', axis=1)
#y = df['preco']
colunas = ['tamanho', 'ano', 'garagem']

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
#linear_rg = LinearRegression()
#linear_rg.fit(X_train, y_train)


############ Modelo Serializado ############

modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME') #'vinicius'
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')#'1234'

basic_auth = BasicAuth(app)


@app.route('/')
def main():
    return 'Minha Primeira API'


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentiment(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt-bt', to='en')
    polaridade = tb_en.sentiment.polarity
    return 'A polaridade da frase é de %s' % polaridade


'''@app.route('/housesprice/<tamanho>')
def previsao_one_var(tamanho):
    y_pred = linear_rg.predict([[float(tamanho)]])
    a = round(y_pred[0][0], 2)
    return str(a)'''


'''@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = linear_rg.predict([dados_input])

    return jsonify(preco=preco[0])'''


@app.route('/cotacao2/', methods=['POST'])
def cotacao2():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])

    return jsonify(preco=preco[0])


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
