#import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify, render_template


class Config(object):
    SECRET_KEY = 'mr_hanky_panky'

#def create_model():
#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.Dense(
#        units=2048,
#        input_shape=(1025, ),
#        activation='relu'
#    ))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(0.2))
#    model.add(tf.keras.layers.Dense(
#        units=256,
#        activation='relu'
#    ))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(0.35))
#    #model.add(tf.keras.layers.Dense(units=16, activation='relu'))
#    model.add(tf.keras.layers.Dense(
#        units=256
#    ))
#    model.add(tf.keras.layers.Dense(
#        units=64,
#        activation='relu'
#    ))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dropout(0.1))
#    model.add(tf.keras.layers.Dense(
#        units=32,
#        activation='relu'
#    ))
#    model.add(tf.keras.layers.BatchNormalization())
#    model.add(tf.keras.layers.Dense(
#        units=1
#    ))
#
#    model.compile(
#        optimizer='rmsprop',
#        loss='mean_squared_error',
#        metrics=['mae',]
#    )
#    return model
#

app = Flask(__name__)
app.config.from_object(Config)


#model = create_model()
#model.load_weights('models/car_prices_weights.h5')

# https://stackoverflow.com/questions/41232105/populate-wtforms-select-field-using-value-selected-from-previous-field/41246506#41246506
# https://stackoverflow.com/questions/46921823/dynamic-choices-wtforms-flask-selectfield

data = pd.read_parquet('data/vehicles_clean.parquet.gzip')
data = data[['manufacturer', 'model']].drop_duplicates()

@app.route('/')
def home():
    user = {'username': 'Nikolaj'}
    return render_template('index.html', user=user)


@app.route('/get_model/<string:brand>')
def get_model(brand):
    models = {}
    for b in data['manufacturer'].unique():
        models[b] = []
        for m in data.loc[data['manufacturer'] == b, 'model'].unique():
            models[b].append(m)
    
    if brand in models:
        return jsonify(sorted(models[brand]))
    else:
        return jsonify([])


@app.route('/get_brands')
def get_brands():
    brands = sorted(data['manufacturer'].unique())
    return jsonify(brands)

#
#
#@app.route('/predict')
#def predict():
#    return render_template('index.html')
#