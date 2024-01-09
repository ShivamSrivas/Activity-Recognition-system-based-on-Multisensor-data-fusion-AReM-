from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        avg_rss12 = float(request.form.get('avg_rss12'))
        var_rss12 = float(request.form.get('var_rss12'))
        avg_rss13 = float(request.form.get('avg_rss13'))
        var_rss13 = float(request.form.get('var_rss13'))
        avg_rss23 = float(request.form.get('avg_rss23'))
        var_rss23 = float(request.form.get('var_rss23'))

        # Provide the correct absolute path for the model file
        model_file_path = r'C:\Users\hp\Desktop\Activity-Recognition-system-based-on-Multisensor-data-fusion-AReM-\model\savedModel\saved_Model.pkl'

        # Load the model
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)

        scaled_input = StandardScaler()
        scaled_input=scaled_input.fit_transform([[avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23]])

        # Make predictions
        prediction = model.predict(scaled_input)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
