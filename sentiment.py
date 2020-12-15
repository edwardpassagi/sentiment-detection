from app import app
from flask import request, render_template, redirect

import src.sentiment_detection as sentiment_detection

@app.route('/')
def home_load():
    return render_template("home.html")


# USAGE
# for i in range(1):
#     reviewInput = input("Insert comment here: ")
    
#     result = naiveBayes(reviewInput, unigVal)
#     printOutput(result)


if __name__ == "__main__":
    app.run()