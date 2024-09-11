from flask import Flask
print('app')
app = Flask(__name__)

from app_ml import routes