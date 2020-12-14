from app import app
from flask import request, render_template, redirect

import src.sentiment_detection as sentiment_detection

@app.route('/')
def home_load():
    return render_template("home.html")


if __name__ == "__main__":
    app.run()