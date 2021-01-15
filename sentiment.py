from app import app
from flask import request, render_template, redirect

import src.sentiment_detection as sentiment_detection

# load home for the first time
@app.route('/')
def home_load():
    return render_template("home.html")

@app.route('/result', methods=['POST'])
def load_result():
    comment_request = request.form['userCommentInput']
    # DEAL WITH RESULT
    result = int(comment_request)
    return render_template("home.html", result=result)


# USAGE
# for i in range(1):
#     reviewInput = input("Insert comment here: ")
    
#     result = naiveBayes(reviewInput, unigVal)
#     printOutput(result)


if __name__ == "__main__":
    app.run()