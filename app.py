from flask import Flask, render_template, request
import joblib

# ✅ FIRST define app
app = Flask(__name__)

# Load model
clf = joblib.load('pickle.pkl')
cv = joblib.load('transform.pkl')


@app.route('/')
def home():
    return render_template('home.html')


# ✅ THEN define route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]

        vect = cv.transform(data).toarray()

        # probabilities
        probs = clf.predict_proba(vect)
        confidence = max(probs[0])

        prediction = clf.predict(vect)[0]

        threshold = 0.65

        if confidence < threshold:
            return render_template(
                'result.html',
                prediction=0,
                unknown=True,
                confidence=round(confidence * 100, 2)
            )
        else:
            return render_template(
                'result.html',
                prediction=prediction,
                unknown=False,
                confidence=round(confidence * 100, 2)
            )


if __name__ == '__main__':
    app.run(debug=True)